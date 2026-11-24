import { isRecord, type AcpElicitationRequestParams } from "../types.js"

export interface OpenCodeQuestion {
  readonly question: string
  readonly header: string
  readonly options: ReadonlyArray<{ readonly label: string; readonly description: string }>
  readonly multiple?: boolean
}

export type AcpElicitationContent = Record<string, string | string[]>

export interface BridgedQuestions {
  readonly questions: readonly OpenCodeQuestion[]
  readonly respond: (
    answers: ReadonlyArray<readonly string[] | undefined>,
  ) => AcpElicitationContent | undefined
}

interface Choice {
  readonly label: string
  readonly value: string
}

interface QuestionPlan {
  readonly question: string
  readonly choiceKey?: string
  readonly textKey?: string
  readonly choices: readonly Choice[]
  readonly multiple: boolean
}

const QUESTION_KEY = /^q(\d+)(_other)?$/
const ANSWER_PREFIX = "User has answered your questions: "
const ANSWER_SUFFIX = ". You can now continue with the user's answers in mind."
const UNANSWERED = "Unanswered"

function nonEmpty(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined
}

function readChoices(schema: Record<string, unknown> | undefined): Choice[] | undefined {
  const variants = Array.isArray(schema?.oneOf)
    ? schema.oneOf
    : Array.isArray(schema?.anyOf)
      ? schema.anyOf
      : undefined
  if (!variants || variants.length === 0) return undefined
  const seen = new Set<string>()
  const choices: Choice[] = []
  for (const variant of variants) {
    const rec = isRecord(variant) ? variant : undefined
    const value = rec?.const
    if (typeof value !== "string") return undefined
    const label = nonEmpty(rec?.title) ?? value
    let unique = label
    for (let n = 2; seen.has(unique); n++) unique = `${label} (${n})`
    seen.add(unique)
    choices.push({ label: unique, value })
  }
  return choices
}

function planQuestion(
  n: number,
  choice?: Record<string, unknown>,
  other?: Record<string, unknown>,
): QuestionPlan | undefined {
  const textKey = other?.type === "string" ? `q${n}_other` : undefined
  if (!choice) {
    return textKey
      ? { question: nonEmpty(other?.title) ?? textKey, textKey, choices: [], multiple: false }
      : undefined
  }
  const multiple = choice.type === "array"
  if (!multiple && choice.type !== "string") return undefined
  const itemsSchema = isRecord(choice.items) ? choice.items : undefined
  const choices = readChoices(multiple ? itemsSchema : choice)
  if (!choices) return undefined
  return {
    question: nonEmpty(choice.title) ?? `q${n}`,
    choiceKey: `q${n}`,
    textKey,
    choices,
    multiple,
  }
}

export function elicitationToQuestions(params: AcpElicitationRequestParams): BridgedQuestions | undefined {
  if (params.mode !== undefined && params.mode !== "form") return undefined
  const requestedSchema = isRecord(params.requestedSchema) ? params.requestedSchema : undefined
  const properties = isRecord(requestedSchema?.properties) ? requestedSchema.properties : undefined
  if (!properties) return undefined

  const fields = new Map<number, { choice?: Record<string, unknown>; other?: Record<string, unknown> }>()
  for (const [key, value] of Object.entries(properties)) {
    const match = QUESTION_KEY.exec(key)
    if (!match || !isRecord(value)) continue
    const idx = Number(match[1])
    const entry = fields.get(idx) ?? {}
    entry[match[2] ? "other" : "choice"] = value
    fields.set(idx, entry)
  }

  const plans: QuestionPlan[] = []
  for (const [n, { choice, other }] of [...fields].sort(([a], [b]) => a - b)) {
    const plan = planQuestion(n, choice, other)
    if (!plan) return undefined
    plans.push(plan)
  }
  if (plans.length === 0) return undefined

  return {
    questions: plans.map((plan, index) => ({
      header: plans.length > 1 ? `Question ${index + 1}` : "Question",
      question: plan.question,
      options: plan.choices.map((c) => ({ label: c.label, description: "" })),
      ...(plan.multiple ? { multiple: true } : {}),
    })),
    respond: (answers) => {
      const content: AcpElicitationContent = {}
      let answered = false
      plans.forEach((plan, index) => {
        const values: string[] = []
        const typed: string[] = []
        for (const entry of answers[index] ?? []) {
          const choice = plan.choices.find((c) => c.label === entry)
          if (choice) values.push(choice.value)
          else if (entry.trim()) typed.push(entry)
        }
        if (plan.choiceKey && values.length > 0) {
          content[plan.choiceKey] = plan.multiple ? values : values[0]
          answered = true
        }
        if (plan.textKey && typed.length > 0) {
          content[plan.textKey] = typed.join(", ")
          answered = true
        }
      })
      return answered ? content : undefined
    },
  }
}

function dequote(value: string): string {
  return value.replace(/\\"/g, '"').replace(/^"/, "").replace(/"$/, "")
}

function matchHead(body: string, position: number, question: string): number | undefined {
  const strict = `"${question}"="`
  if (body.startsWith(strict, position)) return position + strict.length
  const marker = body.indexOf('="', position)
  if (marker < 0) return undefined
  if (dequote(body.slice(position, marker)) !== question) return undefined
  return marker + 2
}

function findSeparator(body: string, from: number, nextQuestion: string): number | undefined {
  const exact = body.indexOf(`", "${nextQuestion}"="`, from)
  if (exact >= 0) return exact
  let at = body.indexOf('", "', from)
  while (at >= 0) {
    const marker = body.indexOf('="', at + 3)
    if (marker < 0) return undefined
    if (dequote(body.slice(at + 3, marker)) === nextQuestion) return at
    at = body.indexOf('", "', at + 1)
  }
  return undefined
}

function splitAnswer(raw: string, question: OpenCodeQuestion): string[] {
  const labels = new Set(question.options.map((o) => o.label))
  if (labels.has(raw)) return [raw]
  if (raw === UNANSWERED || !raw) return []
  if (!question.multiple) return [raw]

  const tokens = raw.split(", ")
  const result: string[] = []
  let typed: string[] = []
  for (let i = 0; i < tokens.length; ) {
    let matched = 0
    for (let j = tokens.length; j > i; j--) {
      if (labels.has(tokens.slice(i, j).join(", "))) {
        matched = j - i
        break
      }
    }
    if (matched === 0) {
      typed.push(tokens[i++])
      continue
    }
    if (typed.length > 0) result.push(typed.join(", "))
    typed = []
    result.push(tokens.slice(i, i + matched).join(", "))
    i += matched
  }
  if (typed.length > 0) result.push(typed.join(", "))
  return result
}

export function parseQuestionToolAnswers(
  text: string,
  questions: readonly OpenCodeQuestion[],
): Array<string[] | undefined> | undefined {
  if (!text.startsWith(ANSWER_PREFIX) || !text.endsWith(ANSWER_SUFFIX)) return undefined
  const body = text.slice(ANSWER_PREFIX.length, text.length - ANSWER_SUFFIX.length)
  const answers: Array<string[] | undefined> = []
  let position = 0
  for (let i = 0; i < questions.length; i++) {
    const head = matchHead(body, position, questions[i].question)
    if (head === undefined) return undefined
    position = head
    let raw: string
    if (i + 1 < questions.length) {
      const end = findSeparator(body, position, questions[i + 1].question)
      if (end === undefined) return undefined
      raw = body.slice(position, end)
      position = end + 3
    } else {
      if (!body.endsWith('"') || body.length - 1 < position) return undefined
      raw = body.slice(position, body.length - 1)
    }
    answers.push(splitAnswer(raw, questions[i]))
  }
  return answers
}
