
import { LlmChunk } from '../../src/types/llm'
import { vi, expect, test, beforeEach } from 'vitest'
import { Plugin2 } from '../mocks/plugins'
import Message from '../../src/models/message'
import OpenAI from '../../src/providers/openai'
import * as _openai from 'openai'

Plugin2.prototype.execute = vi.fn((): Promise<string> => Promise.resolve('result2'))

vi.mock('openai', async() => {
  let streamIteration = 0
  const OpenAI = vi.fn((opts: _openai.ClientOptions) => {
    OpenAI.prototype.apiKey = opts.apiKey
    OpenAI.prototype.baseURL = opts.baseURL
  })
  OpenAI.prototype.chat = {
    completions: {
      create: vi.fn((opts) => {
        if (opts.stream) {
          // alternate: tool-call chunks on the first stream of each
          // generation, then the text response on the follow-up stream
          const iteration = streamIteration++
          return {
            async * [Symbol.asyncIterator]() {
              // first we yield tool call chunks: one start and two argument deltas
              if (iteration % 2 == 0) {
                yield { choices: [{ delta: { tool_calls: [ { id: 1, function: { name: 'plugin2', arguments: '[ "' }} ] }, finish_reason: 'none' } ] }
                yield { choices: [{ delta: { tool_calls: [ { function: { arguments: [ 'ar' ] } }] }, finish_reason: 'none' } ] }
                yield { choices: [{ delta: { tool_calls: [ { function: { arguments: [ 'g" ]' ] } }] }, finish_reason: 'none' } ] }
                yield { choices: [{ finish_reason: 'tool_calls' } ] }
              } else {
                yield { choices: [{ delta: { content: 'response' }, finish_reason: 'none' }] }
                yield { choices: [{ delta: { content: '' }, finish_reason: 'stop' }] }
              }
            },
            controller: {
              abort: vi.fn()
            }
          }
        }
        else {
          return { choices: [{ message: { content: 'response' } }] }
        }
      })
    }
  }
  return { default: OpenAI }
})

beforeEach(() => {
  vi.clearAllMocks()
})

test('Tool argument deltas emit heartbeats when toolCallHeartbeats is enabled', async () => {
  const openai = new OpenAI({ apiKey: '123', toolCallHeartbeats: true })
  openai.addPlugin(new Plugin2())
  const messages = [
    new Message('system', 'instructions'),
    new Message('user', 'prompt1'),
  ]
  const toolCalls: LlmChunk[] = []
  for await (const chunk of openai.generate(openai.buildModel('model'), messages)) {
    if (chunk.type == 'tool') toolCalls.push(chunk)
  }

  // Plugin2's preparation description is constant, so no delta produces a
  // status update: each argument delta must emit a marked heartbeat instead
  const heartbeats = toolCalls.filter((c: any) => c.heartbeat === true)
  expect(heartbeats).toHaveLength(2)
  for (const heartbeat of heartbeats) {
    expect(heartbeat).toMatchObject({ type: 'tool', id: 1, name: 'plugin2', state: 'preparing', status: 'prep2', done: false })
  }

  // the regular chunk sequence is unchanged
  const regular = toolCalls.filter((c: any) => c.heartbeat !== true)
  expect(regular[0]).toStrictEqual({ type: 'tool', id: 1, name: 'plugin2', state: 'preparing', status: 'prep2', done: false })
  expect(regular[1]).toStrictEqual({ type: 'tool', id: 1, name: 'plugin2', state: 'running', status: 'run2', call: { params: ['arg'], result: undefined }, done: false })
  expect(regular[2]).toStrictEqual({ type: 'tool', id: 1, name: 'plugin2', state: 'completed', call: { params: ['arg'], result: 'result2' }, status: undefined, done: true })
})

test('Tool argument deltas emit no heartbeats by default', async () => {
  const openai = new OpenAI({ apiKey: '123' })
  openai.addPlugin(new Plugin2())
  const messages = [
    new Message('system', 'instructions'),
    new Message('user', 'prompt1'),
  ]
  const toolCalls: LlmChunk[] = []
  for await (const chunk of openai.generate(openai.buildModel('model'), messages)) {
    if (chunk.type == 'tool') toolCalls.push(chunk)
  }

  // tool calls flowed but none was a heartbeat
  expect(toolCalls.length).toBeGreaterThan(0)
  expect(toolCalls.filter((c: any) => c.heartbeat === true)).toHaveLength(0)
})
