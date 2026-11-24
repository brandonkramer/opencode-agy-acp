import { describe, expect, test } from "bun:test"
import { elicitationToQuestions, parseQuestionToolAnswers } from "../src/acp/question.js"

describe("question bridge", () => {
  test("sorts q10 after q2, deduplicates labels, and ignores unknown fields", () => {
    const bridged = elicitationToQuestions({
      mode: "form",
      requestedSchema: {
        type: "object",
        properties: {
          unrelatedMeta: { type: "string" },
          q10: {
            type: "string",
            title: "Tenth",
            oneOf: [
              { const: "a", title: "Same" },
              { const: "b", title: "Same" },
            ],
          },
          q2: {
            type: "array",
            title: 'Pick "targets", please',
            items: {
              anyOf: [
                { const: "m", title: "Mac, local" },
                { const: "l", title: "Linux" },
              ],
            },
          },
          q2_other: { type: "string", title: "Other" },
        },
      },
    })

    expect(bridged).toBeDefined()
    expect(bridged!.questions.map((q) => q.question)).toEqual([
      'Pick "targets", please',
      "Tenth",
    ])
    expect(bridged!.questions[1].options.map((o) => o.label)).toEqual(["Same", "Same (2)"])

    const text =
      'User has answered your questions: "Pick \\"targets\\", please"="Mac, local, custom note", "Tenth"="Same (2)". You can now continue with the user\'s answers in mind.'
    const parsed = parseQuestionToolAnswers(text, bridged!.questions)
    expect(parsed).toEqual([["Mac, local", "custom note"], ["Same (2)"]])
    expect(bridged!.respond(parsed!)).toEqual({
      q2: ["m"],
      q2_other: "custom note",
      q10: "b",
    })
  })

  test("returns undefined when nothing was answered so elicitation cancels", () => {
    const bridged = elicitationToQuestions({
      mode: "form",
      requestedSchema: {
        type: "object",
        properties: {
          q1: {
            type: "string",
            title: "Continue?",
            oneOf: [{ const: "yes", title: "Yes" }],
          },
        },
      },
    })!
    expect(bridged.respond([[]])).toBeUndefined()
    expect(parseQuestionToolAnswers("not a question result", bridged.questions)).toBeUndefined()
  })
})
