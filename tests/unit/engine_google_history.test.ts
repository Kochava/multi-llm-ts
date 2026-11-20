import { expect, test } from 'vitest'
import Message from '../../src/models/message'
import Google from '../../src/providers/google'
import { LlmToolCall } from '../../src/types/llm'

// A chat's history reaches Gemini as one assistant message per reply, holding every tool
// call of every round of that reply (often run by another engine). Vertex refuses a
// history where a function call turn is not answered by one turn with the same number of
// function responses, where turns do not alternate, or where a function response is not
// an object; any of these failed every later turn of the chat with a 400.

const google = () => new Google({ apiKey: '123' })
const model = () => google().buildModel('gemini-4-hydrogen')

const call = (id: string, fn: string, args: any, result: any): LlmToolCall => ({ id, function: fn, args, result })

const reply = () => new Message('assistant', 'The total is 42.', undefined, [
  call('toolu_1', 'list_tables', { db: 'mmp' }, { tables: ['traffic'] }),
  call('toolu_2', 'describe_table', { table: 'traffic' }, { columns: ['day', 'total'] }),
  call('toolu_3', 'run_query', { sql: 'select 1' }, { error: 'syntax error' }),
])

const thread = () => [
  new Message('system', 'instructions'),
  new Message('user', 'total for September'),
  reply(),
  new Message('user', 'and for October?'),
]

test('every function response of a reply travels in the one user turn that answers its calls', () => {
  const payload = google().buildGooglePayload(thread(), model())
  const callTurn = payload.findIndex((c) => c.parts!.some((p) => p.functionCall))
  const calls = payload[callTurn].parts!.filter((p) => p.functionCall)
  const answer = payload[callTurn + 1]
  expect(answer.role).toBe('user')
  expect(answer.parts!.map((p) => p.functionResponse?.name)).toStrictEqual(calls.map((p) => p.functionCall!.name))
  // and no other turn carries a function response
  expect(payload.filter((c) => c.parts!.some((p) => p.functionResponse))).toHaveLength(1)
})

test('turns alternate between user and model', () => {
  const payload = google().buildGooglePayload(thread(), model())
  for (let i = 1; i < payload.length; i++) {
    expect(payload[i].role, `turn ${i}`).not.toBe(payload[i - 1].role)
  }
  expect(payload.map((c) => c.role).every((r) => r === 'user' || r === 'model')).toBe(true)
})

test('what the model wrote after its tools comes after their results', () => {
  const payload = google().buildGooglePayload(thread(), model())
  const answer = payload.findIndex((c) => c.parts!.some((p) => p.functionResponse))
  expect(payload[answer + 1]).toStrictEqual({ role: 'model', parts: [{ text: 'The total is 42.' }] })
  expect(payload[answer + 2]).toStrictEqual({ role: 'user', parts: [{ text: 'and for October?' }] })
})

test('a function response is always an object', () => {
  const message = new Message('assistant', 'done', undefined, [
    call('a', 'text_tool', {}, 'plain text'),
    call('b', 'rows_tool', {}, [{ total: 42 }]),
    call('c', 'number_tool', {}, 42),
    call('d', 'object_tool', {}, { ok: true }),
  ])
  const payload = google().buildGooglePayload([new Message('user', 'go'), message], model())
  const responses = payload.flatMap((c) => c.parts!.filter((p) => p.functionResponse).map((p) => p.functionResponse!.response))
  expect(responses).toStrictEqual([
    { result: 'plain text' },
    { result: [{ total: 42 }] },
    { result: 42 },
    { ok: true },
  ])
})

test('a tool call without args is sent with empty args instead of failing the turn', () => {
  // a call interrupted before its args arrived is stored without them
  const message = new Message('assistant', 'stopped', undefined, [
    call('a', 'run_query', undefined, { error: 'interrupted' }),
    call('b', 'run_query', null, { error: 'interrupted' }),
    { id: 'c', function: 'run_query', args: '', result: { error: 'interrupted' } } as LlmToolCall,
  ])
  const payload = google().buildGooglePayload([new Message('user', 'go'), message], model())
  const args = payload.flatMap((c) => c.parts!.filter((p) => p.functionCall).map((p) => p.functionCall!.args))
  expect(args).toStrictEqual([{}, {}, {}])
})

test('empty text parts are not sent', () => {
  // a reply that only called tools has no text of its own
  const message = new Message('assistant', '', undefined, [call('a', 'list_tables', {}, { tables: [] })])
  const payload = google().buildGooglePayload([new Message('user', 'go'), message, new Message('user', 'next')], model())
  for (const content of payload) {
    expect(content.parts!.length).toBeGreaterThan(0)
    for (const part of content.parts!) {
      if ('text' in part) expect(part.text).not.toBe('')
    }
  }
  // the function responses and the next prompt share the user turn
  expect(payload.map((c) => c.role)).toStrictEqual(['user', 'model', 'user'])
})

test('consecutive messages of the same role are merged', () => {
  const payload = google().buildGooglePayload([
    new Message('user', 'first'),
    new Message('user', 'second'),
    new Message('assistant', 'a'),
    new Message('assistant', 'b'),
  ], model())
  expect(payload).toStrictEqual([
    { role: 'user', parts: [{ text: 'first' }, { text: 'second' }] },
    { role: 'model', parts: [{ text: 'a' }, { text: 'b' }] },
  ])
})

test('a reply with thought signatures keeps them on its function calls', () => {
  const message = new Message('assistant', 'ok', undefined, [
    { ...call('a', 'list_tables', {}, { tables: [] }), thoughtSignature: 'sig' },
  ])
  const payload = google().buildGooglePayload([new Message('user', 'go'), message], model())
  expect(payload[1]).toStrictEqual({ role: 'model', parts: [{ functionCall: { name: 'list_tables', args: {} }, thoughtSignature: 'sig' }] })
})

test('the engine never sends undefined tool call arguments to any provider', () => {
  const message = new Message('assistant', 'stopped', undefined, [
    call('a', 'run_query', undefined, { error: 'interrupted' }),
    call('b', 'run_query', null, { error: 'interrupted' }),
  ])
  const payload = google().buildPayload(model(), [new Message('user', 'go'), message]) as any[]
  expect(payload[1].tool_calls.map((tc: any) => tc.function.arguments)).toStrictEqual(['{}', '{}'])
})
