import { PluginParameter, PluginTool } from './types/plugin'
import { LlmTool, LlmToolArrayItems, LlmToolOpenAI, LlmToolParameterOpenAI } from './types/llm'

/**
 * Type guard to check if a tool is in the new PluginTool format.
 * PluginTool has parameters as an array, while OpenAI format has nested structure.
 */
export function isToolDefinition(tool: LlmTool): tool is PluginTool {
  return Array.isArray((tool as PluginTool).parameters)
}

/**
 * Type guard to check if a tool is in the legacy OpenAI format.
 */
export function isLegacyOpenAITool(tool: LlmTool): tool is LlmToolOpenAI {
  return 'type' in tool && tool.type === 'function' && 'function' in tool
}

/**
 * Normalizes any LlmTool format to PluginTool.
 * Use this to convert legacy OpenAI format tools to the new format.
 */
export function normalizeToToolDefinition(tool: LlmTool): PluginTool {
  if (isToolDefinition(tool)) {
    return tool
  }

  // Convert from OpenAI format
  const params: PluginParameter[] = []
  const props = tool.function.parameters?.properties || {}
  const required = tool.function.parameters?.required || []

  for (const [name, prop] of Object.entries(props)) {
    params.push({
      name,
      type: prop.type,
      description: prop.description,
      required: required.includes(name),
      ...(prop.enum ? { enum: prop.enum } : {}),
      ...(prop.items ? { items: convertOpenAIItems(prop.items) } : {}),
    })
  }

  return {
    name: tool.function.name,
    description: tool.function.description,
    parameters: params,
  }
}

/**
 * Converts a PluginTool to OpenAI format.
 * Use this for providers that use OpenAI SDK or expect OpenAI format.
 */
export function toolDefinitionToOpenAI(tool: PluginTool): LlmToolOpenAI {
  const properties: Record<string, LlmToolParameterOpenAI> = {}
  const required: string[] = []

  for (const param of tool.parameters) {
    const type = param.type || (param.items ? 'array' : 'string')
    const prop: LlmToolParameterOpenAI = {
      type,
      description: param.description,
      ...(param.enum ? { enum: param.enum } : {}),
    }
    // arrays must have items — default to string if missing
    if (type === 'array') {
      prop.items = convertItemsToJsonSchema(param.items)
    }
    properties[param.name] = prop
    if (param.required) {
      required.push(param.name)
    }
  }

  return {
    type: 'function',
    function: {
      name: tool.name,
      description: tool.description,
      parameters: {
        type: 'object',
        properties,
        required,
      },
    },
  }
}

/**
 * Converts any LlmTool to OpenAI format.
 * If already in OpenAI format, returns as-is.
 */
export function toOpenAITool(tool: LlmTool): LlmToolOpenAI {
  if (isLegacyOpenAITool(tool)) {
    return tool
  }
  return toolDefinitionToOpenAI(tool)
}

/**
 * Normalizes an array of tools to PluginTool format.
 */
export function normalizeTools(tools: LlmTool[]): PluginTool[] {
  return tools.map(normalizeToToolDefinition)
}

/**
 * Converts an array of tools to OpenAI format.
 */
export function toOpenAITools(tools: LlmTool[]): LlmToolOpenAI[] {
  return tools.map(toOpenAITool)
}

// Helper to convert OpenAI items format to PluginParameter items format
function convertOpenAIItems(items: LlmToolArrayItems): PluginParameter['items'] {
  if (!items.properties) {
    return { type: items.type }
  }

  // LlmToolArrayItems.properties is an array of LlmToolArrayItem
  const props: PluginParameter[] = items.properties.map((prop) => ({
    name: prop.name,
    type: prop.type,
    description: prop.description,
    required: prop.required,
  }))

  return {
    type: items.type,
    properties: props,
  }
}

// Helper to convert a single PluginParameter to a JSON Schema property
export function pluginParamToJsonSchema(param: PluginParameter): any {
  const type = param.type || (param.items ? 'array' : 'string')
  const prop: any = {
    type,
    description: param.description,
    ...(param.enum ? { enum: param.enum } : {}),
  }
  if (type === 'array') {
    prop.items = convertItemsToJsonSchema(param.items)
  }
  return prop
}

// Helper to convert PluginParameter items to JSON Schema items format
// Returns a proper JSON Schema object with properties as Record<string, ...>
function convertItemsToJsonSchema(items: PluginParameter['items']): any {
  if (!items) return { type: 'string' }
  if (!items.properties) {
    // array-of-arrays: propagate nested items
    if (items.type === 'array' && items.items) {
      return { type: 'array', items: { type: items.items.type || 'string' } }
    }
    return { type: items.type }
  }

  // object with nested properties: convert array to Record
  const props: Record<string, any> = {}
  const required: string[] = []
  for (const param of items.properties) {
    props[param.name] = pluginParamToJsonSchema(param)
    if (param.required) {
      required.push(param.name)
    }
  }

  return {
    type: items.type || 'object',
    properties: props,
    required,
  }
}

/**
 * Drops the values a strict-all-required model had to send for optional fields
 * it did not set, walking the tool's declared schema through nested objects and
 * array items. Returns new objects; the input is left untouched.
 *
 * Under the contract:
 * - null on an optional field is dropped
 * - '' on an optional field is dropped only when the field declares a format or
 *   an enum without '', i.e. where '' is provably invalid; elsewhere it may be a
 *   deliberate blank (e.g. clearing a description) and is kept
 * - required fields keep whatever was sent so the tool can report it
 *
 * Optional means `required: false` on a PluginParameter, or absent from the
 * `required` list of a JSON Schema object — the same fields the OpenAI Responses
 * conversion makes nullable.
 */
export function stripUnsetOptionalArgs(tool: PluginTool, args: any): any {
  return stripUnsetInObject(declaredFields({ properties: tool.parameters }), args)
}

type DeclaredField = { schema: any, optional: boolean }

function stripUnsetInValue(schema: any, value: any): any {
  if (value === null || typeof value !== 'object') return value
  const shape = structuralSchema(schema)
  if (!shape) return value
  if (Array.isArray(value)) {
    return shape.items ? value.map((entry) => stripUnsetInValue(shape.items, entry)) : value
  }
  return stripUnsetInObject(declaredFields(shape), value)
}

function stripUnsetInObject(fields: Record<string, DeclaredField> | undefined, value: any): any {
  if (!fields || value === null || typeof value !== 'object' || Array.isArray(value)) return value
  const result: Record<string, any> = {}
  for (const [key, child] of Object.entries(value)) {
    const field = fields[key]
    if (field?.optional && isUnset(field.schema, child)) continue
    result[key] = field ? stripUnsetInValue(field.schema, child) : child
  }
  return result
}

// properties come either as PluginParameter[] or as a JSON Schema record
function declaredFields(schema: any): Record<string, DeclaredField> | undefined {
  const properties = schema.properties
  if (Array.isArray(properties)) {
    return Object.fromEntries(properties.map((param: PluginParameter) => [param.name, { schema: param, optional: param.required === false }]))
  }
  if (properties && typeof properties === 'object') {
    const required = Array.isArray(schema.required) ? schema.required : undefined
    return Object.fromEntries(Object.entries(properties).map(([key, child]) => [key, { schema: child, optional: !!required && !required.includes(key) }]))
  }
  return undefined
}

function isUnset(schema: any, value: any): boolean {
  if (value === null) return true
  if (value !== '' || !schema) return false
  return schema.format !== undefined || (Array.isArray(schema.enum) && !schema.enum.includes(''))
}

// the object/array shape may sit in a union branch (anyOf, oneOf, or a type list)
function structuralSchema(schema: any): any {
  if (!schema || typeof schema !== 'object') return undefined
  if (schema.properties || schema.items) return schema
  const branches = [
    ...(Array.isArray(schema.anyOf) ? schema.anyOf : []),
    ...(Array.isArray(schema.oneOf) ? schema.oneOf : []),
    ...(Array.isArray(schema.type) ? schema.type : []),
  ]
  return branches.find((entry: any) => entry && typeof entry === 'object' && (entry.properties || entry.items))
}
