import { vi, beforeEach, expect, test, describe } from 'vitest'
import { Plugin2, PluginUpdate } from '../mocks/plugins'
import Message from '../../src/models/message'
import OpenAI from '../../src/providers/openai'
import { LlmChunk, LlmChunkTool, LlmToolCallGuard } from '../../src/types/llm'
import * as _openai from 'openai'

// the tool the mocked model calls on its first turn
let modelToolCall = { name: 'plugin2', arguments: '{}' }
let callCount = 0

vi.mock('openai', async () => {
  const OpenAI = vi.fn((opts: _openai.ClientOptions) => {
    OpenAI.prototype.apiKey = opts.apiKey
    OpenAI.prototype.baseURL = opts.baseURL
  })
  OpenAI.prototype.chat = {
    completions: {
      create: vi.fn(() => {
        callCount++
        const first = callCount === 1
        return {
          async * [Symbol.asyncIterator]() {
            if (first) {
              yield { choices: [{ delta: { tool_calls: [{ id: 'tool_1', function: modelToolCall }] }, finish_reason: 'none' }] }
              yield { choices: [{ finish_reason: 'tool_calls' }] }
            } else {
              yield { choices: [{ delta: { content: 'done' }, finish_reason: 'none' }] }
              yield { choices: [{ delta: { content: '' }, finish_reason: 'stop' }] }
            }
          },
          controller: { abort: vi.fn() },
        }
      }),
    },
  }
  return { default: OpenAI }
})

const run = async (guard: LlmToolCallGuard | undefined, opts: Record<string, unknown> = {}) => {
  const openai = new OpenAI({ apiKey: '123' })
  openai.addPlugin(new Plugin2())
  openai.addPlugin(new PluginUpdate())
  const chunks: LlmChunk[] = []
  for await (const chunk of openai.generate(openai.buildModel('model'), [new Message('system', 's'), new Message('user', 'u')], { toolCallGuard: guard, ...opts })) {
    chunks.push(chunk)
  }
  const create = (_openai.default as any).prototype.chat.completions.create
  // what the model received as the tool's result on its second turn
  const toolMessage = create.mock.calls[1]?.[0].messages.find((m: any) => m.role === 'tool')
  const completed = chunks.filter((c): c is LlmChunkTool => c.type === 'tool' && c.done).pop()
  return { toolMessage, completed }
}

beforeEach(() => {
  vi.clearAllMocks()
  callCount = 0
  modelToolCall = { name: 'plugin2', arguments: '{}' }
  Plugin2.prototype.execute = vi.fn((): Promise<any> => Promise.resolve({ owner: 'john@example.com', rows: ['jane@example.com', 3] }))
})

describe('toolCallGuard', () => {

  test('the model and the stored call see the result afterExecute returns', async () => {
    const afterExecute = vi.fn(async () => ({ owner: '[EMAIL]', rows: ['[EMAIL]', 3] }))
    const { toolMessage, completed } = await run({ afterExecute })

    expect(afterExecute).toHaveBeenCalledWith(expect.objectContaining({ model: 'model' }), 'plugin2', {}, { owner: 'john@example.com', rows: ['jane@example.com', 3] })
    expect(toolMessage.content).toContain('[EMAIL]')
    expect(toolMessage.content).not.toContain('example.com')
    expect(completed.call.result).toEqual({ owner: '[EMAIL]', rows: ['[EMAIL]', 3] })
  })

  test('beforeExecute rewrites the arguments the tool and the validator receive', async () => {
    modelToolCall = { name: 'plugin2', arguments: '{"to":"john@example.com"}' }
    const validator = vi.fn().mockResolvedValue({ decision: 'allow' })
    await run({ beforeExecute: async () => ({ args: { to: '[EMAIL]' } }) }, { toolExecutionValidation: validator })

    expect(validator).toHaveBeenCalledWith(expect.anything(), 'plugin2', { to: '[EMAIL]' })
    expect(Plugin2.prototype.execute).toHaveBeenCalledWith(expect.anything(), { to: '[EMAIL]' })
  })

  test('beforeExecute can refuse a call: the tool never runs and the model gets the reason', async () => {
    const { toolMessage } = await run({ beforeExecute: async () => ({ error: 'Tool not run: its arguments contain personal data (email)' }) })

    expect(Plugin2.prototype.execute).not.toHaveBeenCalled()
    expect(toolMessage.content).toContain('its arguments contain personal data (email)')
  })

  test('a throwing beforeExecute does not run the tool', async () => {
    const { toolMessage } = await run({ beforeExecute: async () => { throw new Error('boom') } })

    expect(Plugin2.prototype.execute).not.toHaveBeenCalled()
    expect(toolMessage.content).toContain('its arguments could not be checked')
  })

  test('a throwing afterExecute withholds the result instead of passing it through', async () => {
    const { toolMessage, completed } = await run({ afterExecute: async () => { throw new Error('boom') } })

    expect(toolMessage.content).not.toContain('example.com')
    expect(toolMessage.content).toContain('result was withheld')
    expect(JSON.stringify(completed.call.result)).not.toContain('example.com')
  })

  test('guards a delegated tool call the same way', async () => {
    modelToolCall = { name: 'mcp_lookup', arguments: '{}' }
    const delegate = { getTools: () => [], execute: vi.fn(async () => ({ email: 'john@example.com' })) }
    const { toolMessage } = await run({ afterExecute: async () => ({ email: '[EMAIL]' }) }, { toolExecutionDelegate: delegate })

    expect(delegate.execute).toHaveBeenCalled()
    expect(toolMessage.content).toContain('[EMAIL]')
    expect(toolMessage.content).not.toContain('example.com')
  })

  test('applies afterExecute to the final result only, not to status updates', async () => {
    modelToolCall = { name: 'pluginUpdate', arguments: '{}' }
    const afterExecute = vi.fn(async (_c: unknown, _t: string, _a: unknown, result: unknown) => `checked ${result}`)
    const { toolMessage } = await run({ afterExecute })

    expect(afterExecute).toHaveBeenCalledTimes(1)
    expect(afterExecute.mock.calls[0][3]).toBe('result')
    expect(toolMessage.content).toContain('checked result')
  })

  test('leaves everything as it was without a guard', async () => {
    const { toolMessage } = await run(undefined)
    expect(toolMessage.content).toContain('john@example.com')
  })

})
