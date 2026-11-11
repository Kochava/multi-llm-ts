
import { LlmToolExecutionValidationResponse, ToolParameterType } from './llm'

export interface IPlugin {

  serializeInTools(): boolean
  isEnabled(): boolean
  getName(): string
  getDescription(): string
  getPreparationDescription(tool: string, partialArgs?: any): string
  getRunningDescription(tool: string, args: any): string
  getCompletedDescription(tool: string, args: any, results: any): string|undefined
  getParameters(): PluginParameter[]
  execute(context: PluginExecutionContext , parameters: any): Promise<any>
  executeWithUpdates?(context: PluginExecutionContext , parameters: any): AsyncGenerator<PluginExecutionUpdate>
}

export type PluginParameter = {
  name: string
  type: ToolParameterType
  description: string
  required?: boolean
  enum?: string[]
  items?: {
    type: string,
    properties?: PluginParameter[]
    items?: { type: string }
  }
}

/**
 * Provider-agnostic tool definition format.
 * This is the recommended format for defining tools in plugins.
 * It will be converted to provider-specific formats internally.
 */
export type PluginTool = {
  name: string
  description: string
  parameters: PluginParameter[]
}

/**
 * How a provider coerced the tool schema it sent to the model.
 * - 'strict-all-required': every property was sent as required and the optional
 *   ones as nullable, so the model had to send a value (null, sometimes '') for
 *   optionals it did not mean to set. The engine drops those before the tool runs.
 */
export type ToolArgsCoercion = 'strict-all-required'

export type PluginExecutionContext = {
  model: string
  abortSignal?: AbortSignal
  coercion?: ToolArgsCoercion
}

export type PluginExecutionStatusUpdate = {
  type: 'status'
  status: string
}

export type PluginExecutionResult = {
  type: 'result'
  result: any
  canceled?: boolean
  validation?: LlmToolExecutionValidationResponse
}

export type PluginExecutionUpdate = PluginExecutionStatusUpdate | PluginExecutionResult

/**
 * Delegate for external tool execution.
 * Allows callers to inject tool definitions and handle execution externally
 * without creating plugin classes. Tools from the delegate are per-request,
 * not registered globally on the engine.
 */
export type ToolExecutionDelegate = {
  getTools(): Promise<PluginTool[]> | PluginTool[]
  execute(context: PluginExecutionContext, tool: string, args: any): Promise<any>
}
