import { expect, test, describe } from 'vitest'
import OpenAI from '../../src/providers/openai'
import { MultiToolPlugin, Plugin } from '../../src/plugin'
import { PluginExecutionContext, PluginParameter } from '../../src/types/plugin'

// tool-argument normalization (LlmEngine.normalizeToolArgs): it only applies when
// the provider declares the strict-all-required coercion contract; the provider
// wiring (one normalization shared by chunks, history, guard and tool) is covered
// in engine_openai_responses.test.ts

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

const normalize = async (plugin: Plugin | MultiToolPlugin, context: PluginExecutionContext, tool: string, args: any) => {
  const engine = new OpenAI({ apiKey: '123' })
  engine.addPlugin(plugin)
  const declared = (await (engine as any).getAvailableTools()).find((t: any) => t.name === tool)
  expect(declared).toBeDefined()
  return (engine as any).normalizeToolArgs(context, declared, args)
}

const runLookup = (context: PluginExecutionContext, args: any) => normalize(new LookupPlugin(), context, 'lookup', args)
const runBatch = (args: any) => normalize(new BatchPlugin(), strict, 'create_tasks', args)

describe('normalizeToolArgs', () => {

  test('leaves every null and empty string untouched when no coercion contract is declared', async () => {
    const args = { query: 'scrum', offset: null, team: '', limit: null, status: '', due_on: '', kind: 'bug' }
    const params = await runLookup({ model: 'model' }, structuredClone(args))
    expect(params).toStrictEqual(args)
  })

  test('leaves the arguments untouched when the tool has no declaration', async () => {
    const engine = new OpenAI({ apiKey: '123' })
    const args = { id: 'a', note: null }
    expect((engine as any).normalizeToolArgs(strict, undefined, args)).toStrictEqual({ id: 'a', note: null })
  })

  test('drops nulls for optional params under strict-all-required', async () => {
    const params = await runLookup(strict, { query: 'scrum', offset: null, limit: null, kind: 'bug' })
    expect(params).toStrictEqual({ query: 'scrum', kind: 'bug' })
  })

  test('returns a new object and leaves the parsed arguments untouched', async () => {
    const args = { query: 'scrum', offset: null, kind: 'bug' }
    await runLookup(strict, args)
    expect(args).toStrictEqual({ query: 'scrum', offset: null, kind: 'bug' })
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

  test('drops unset optionals nested inside array items', async () => {
    const params = await runBatch({
      tasks: [{ name: 'Task one', parent: null, due_at: null, approval_status: null }],
    })
    expect(params).toStrictEqual({ tasks: [{ name: 'Task one' }] })
  })

  test('applies the empty-string rule inside array items', async () => {
    const params = await runBatch({
      tasks: [{ name: 'Task one', parent: '', due_at: '', approval_status: '' }],
    })
    expect(params).toStrictEqual({ tasks: [{ name: 'Task one', parent: '' }] })
  })

  test('drops unset optionals inside a nested object', async () => {
    const params = await runBatch({
      tasks: [{ name: 'Task one', assignee: { gid: '123', note: null } }],
    })
    expect(params).toStrictEqual({ tasks: [{ name: 'Task one', assignee: { gid: '123' } }] })
  })

  test('keeps a null on a required nested field so the tool can report it', async () => {
    const params = await runBatch({
      tasks: [{ name: null, parent: null, assignee: { gid: null } }],
    })
    expect(params).toStrictEqual({ tasks: [{ name: null, assignee: { gid: null } }] })
  })

})
