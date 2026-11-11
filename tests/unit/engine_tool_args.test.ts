import { vi, expect, test, describe } from 'vitest'
import OpenAI from '../../src/providers/openai'
import { MultiToolPlugin, Plugin } from '../../src/plugin'
import { PluginExecutionContext, PluginExecutionUpdate, PluginParameter, ToolExecutionDelegate } from '../../src/types/plugin'
import { LlmToolCallGuard } from '../../src/types/llm'

// tool-argument normalization at the engine → plugin boundary (LlmEngine.callTool):
// it only runs when the provider declares the strict-all-required coercion contract

class LookupPlugin extends Plugin {
  getName(): string { return 'lookup' }
  getDescription(): string { return 'look things up' }
  getRunningDescription(): string { return 'running lookup' }
  getParameters(): PluginParameter[] {
    return [
      { name: 'query', type: 'string', description: 'q', required: true },
      { name: 'offset', type: 'string', description: 'pagination token', required: false },
      { name: 'team', type: 'string', description: 'team', required: false },
      { name: 'limit', type: 'number', description: 'limit', required: false },
      { name: 'status', type: 'string', description: 'status', required: false, enum: ['open', 'closed'] },
      { name: 'due_on', type: 'string', description: 'due date', required: false, format: 'date' } as any,
      { name: 'kind', type: 'string', description: 'kind', required: true, enum: ['bug', 'task'] },
    ]
  }
  async execute(): Promise<any> { return { ok: true } }
}

// asana create_tasks in the field: every optional field sits inside array items
class BatchPlugin extends MultiToolPlugin {
  getName(): string { return 'batch' }
  getDescription(): string { return 'batch things' }
  getRunningDescription(): string { return 'running batch' }
  handlesTool(name: string): boolean { return name === 'create_tasks' }
  async getTools(): Promise<any[]> {
    return [{
      name: 'create_tasks',
      description: 'create several tasks',
      parameters: [{
        name: 'tasks',
        type: 'array',
        description: 'the tasks',
        required: true,
        items: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            parent: { type: 'string' },
            due_at: { type: 'string', format: 'date-time' },
            approval_status: { type: 'string', enum: ['pending', 'approved'] },
            assignee: {
              type: 'object',
              properties: { gid: { type: 'string' }, note: { type: 'string' } },
              required: ['gid'],
            },
          },
          required: ['name'],
        },
      }],
    }]
  }
  async execute(): Promise<any> { return { ok: true } }
}

const strict: PluginExecutionContext = { model: 'model', coercion: 'strict-all-required' }

const callTool = async (engine: OpenAI, context: PluginExecutionContext, tool: string, args: any, opts: {
  delegate?: ToolExecutionDelegate, guard?: LlmToolCallGuard, validator?: any
} = {}): Promise<PluginExecutionUpdate[]> => {
  const updates: PluginExecutionUpdate[] = []
  for await (const update of (engine as any).callTool(context, tool, args, opts.delegate, opts.validator, opts.guard)) {
    updates.push(update)
  }
  return updates
}

const runLookup = async (context: PluginExecutionContext, args: any) => {
  const engine = new OpenAI({ apiKey: '123' })
  const plugin = new LookupPlugin()
  const execute = vi.spyOn(plugin, 'execute')
  engine.addPlugin(plugin)
  await callTool(engine, context, 'lookup', args)
  expect(execute).toHaveBeenCalledTimes(1)
  return execute.mock.calls[0][1]
}

const runBatch = async (args: any) => {
  const engine = new OpenAI({ apiKey: '123' })
  const plugin = new BatchPlugin()
  const execute = vi.spyOn(plugin, 'execute')
  engine.addPlugin(plugin)
  await callTool(engine, strict, 'create_tasks', args)
  expect(execute).toHaveBeenCalledTimes(1)
  return execute.mock.calls[0][1] as any
}

describe('callTool argument normalization', () => {

  test('leaves every null and empty string untouched when no coercion contract is declared', async () => {
    const args = { query: 'scrum', offset: null, team: '', limit: null, status: '', due_on: '', kind: 'bug' }
    const params = await runLookup({ model: 'model' }, structuredClone(args))
    expect(params).toStrictEqual(args)
  })

  test('drops nulls for optional params under strict-all-required', async () => {
    const params = await runLookup(strict, { query: 'scrum', offset: null, limit: null, kind: 'bug' })
    expect(params).toStrictEqual({ query: 'scrum', kind: 'bug' })
  })

  test('keeps a null on a required param so the tool can report it', async () => {
    const params = await runLookup(strict, { query: null, offset: null, kind: null })
    expect(params).toStrictEqual({ query: null, kind: null })
  })

  test('drops an empty string only where format or enum makes it invalid', async () => {
    const params = await runLookup(strict, { query: 'scrum', team: '', status: '', due_on: '', kind: '' })
    // team is a plain optional string: '' may be a deliberate blank and stays
    // kind is required: its '' reaches the tool so it can report it
    expect(params).toStrictEqual({ query: 'scrum', team: '', kind: '' })
  })

  test('drops unset optionals nested inside array items, before the multi-tool wrap', async () => {
    const payload = await runBatch({
      tasks: [{ name: 'Task one', parent: null, due_at: null, approval_status: null }],
    })
    expect(payload).toStrictEqual({ tool: 'create_tasks', parameters: { tasks: [{ name: 'Task one' }] } })
  })

  test('applies the empty-string rule inside array items', async () => {
    const payload = await runBatch({
      tasks: [{ name: 'Task one', parent: '', due_at: '', approval_status: '' }],
    })
    expect(payload.parameters).toStrictEqual({ tasks: [{ name: 'Task one', parent: '' }] })
  })

  test('drops unset optionals inside a nested object', async () => {
    const payload = await runBatch({
      tasks: [{ name: 'Task one', assignee: { gid: '123', note: null } }],
    })
    expect(payload.parameters).toStrictEqual({ tasks: [{ name: 'Task one', assignee: { gid: '123' } }] })
  })

  test('keeps a null on a required nested field so the tool can report it', async () => {
    const payload = await runBatch({
      tasks: [{ name: null, parent: null, assignee: { gid: null } }],
    })
    expect(payload.parameters).toStrictEqual({ tasks: [{ name: null, assignee: { gid: null } }] })
  })

  test('the guard and the validator see the normalized arguments', async () => {
    const engine = new OpenAI({ apiKey: '123' })
    engine.addPlugin(new LookupPlugin())
    const beforeExecute = vi.fn(async (_context: any, _tool: string, args: any) => ({ args }))
    const validator = vi.fn().mockResolvedValue({ decision: 'allow' })
    await callTool(engine, strict, 'lookup', { query: 'scrum', offset: null, kind: 'bug' }, { guard: { beforeExecute }, validator })
    expect(beforeExecute.mock.calls[0][2]).toStrictEqual({ query: 'scrum', kind: 'bug' })
    expect(validator.mock.calls[0][2]).toStrictEqual({ query: 'scrum', kind: 'bug' })
  })

  test('normalizes delegate tools against the schema the delegate declares', async () => {
    const engine = new OpenAI({ apiKey: '123' })
    const delegate: ToolExecutionDelegate = {
      getTools: () => [{ name: 'remote', description: 'remote tool', parameters: [
        { name: 'id', type: 'string', description: 'id', required: true },
        { name: 'note', type: 'string', description: 'note', required: false },
      ] }],
      execute: vi.fn(async () => ({ ok: true })),
    }
    await callTool(engine, strict, 'remote', { id: 'a', note: null }, { delegate })
    expect(vi.mocked(delegate.execute).mock.calls[0][2]).toStrictEqual({ id: 'a' })
  })

})
