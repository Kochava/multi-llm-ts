import { Content, Environment, ThinkingConfig, ThinkingLevel, FunctionCallingConfigMode, FunctionDeclaration, FunctionResponse, GenerateContentConfig, GenerateContentResponse, GoogleGenAI, Part, Schema, Type } from '@google/genai'
import { minimatch } from 'minimatch'
import { zodToJsonSchema } from 'zod-to-json-schema'
import LlmEngine from '../engine'
import logger from '../logger'
import Attachment from '../models/attachment'
import Message from '../models/message'
import { Plugin } from '../plugin'
import { ChatModel, EngineCreateOpts, ModelCapabilities, ModelGoogle } from '../types/index'
import { LlmChunk, LlmCompletionOpts, LlmCompletionPayloadContent, LlmCompletionPayloadTool, LLmContentPayloadText, LlmGoogleThinkingLevel, LlmGoogleThinkingStyle, LlmResponse, LlmStream, LlmStreamingContext, LlmStreamingResponse, LlmToolCallInfo, LlmUsage } from '../types/llm'
import { PluginExecutionResult, PluginParameter } from '../types/plugin'
import { addUsages, zeroUsage } from '../usage'

//
// https://ai.google.dev/gemini-api/docs
//

type GoogleCompletionOpts = LlmCompletionOpts & {
  instruction?: string
}

export interface GoogleComputerToolInfo {
  plugin: Plugin
  screenSize(): { width: number, height: number }
  screenNumber(): number
}

export type GoogleStreamingContext = LlmStreamingContext<Content> & {
  opts: GoogleCompletionOpts
  requestUsage: LlmUsage
  textContentBlock?: string
}

export default class GoogleEngine extends LlmEngine {

  client: GoogleGenAI
  computerInfo: GoogleComputerToolInfo|null = null

  constructor(config: EngineCreateOpts, computerInfo: GoogleComputerToolInfo|null = null) {
    super(config)
    this.client = new GoogleGenAI({
      apiKey: config.apiKey!,
      ...(config.baseURL ? { httpOptions: {
        baseUrl: config.baseURL }
      } : {}),
    })
    this.computerInfo = computerInfo
  }

  getId(): string {
    return 'google'
  }

  // https://ai.google.dev/gemini-api/docs/models/gemini

  getModelCapabilities(model: ModelGoogle): ModelCapabilities {

    const visionGlobs = [
      'gemma-3*',
      'gemma-4*',
      'gemini-1.5-pro-latest',
      'gemini-1.5-flash-*',
      'gemini-2.0-flash-*',
      'gemini-exp-1206',
      'gemini-2.0-flash-thinking-*',
      'gemini-2.5-*',
      'gemini-3{-,.}*',
    ]

    const excludeVisionGlobs = [
      'gemma-3-1b*',
      '*tts*',
    ]

    const reasoningGlobs = [
      '*thinking*',
      'gemma-4*',
      'gemini-2.5-flash*',
      'gemini-2.5-pro*',
      'gemini-3{-,.}*',
      'gemini-4{-,.}*',
    ]

    // speech models are swept in by the version globs but do not reason
    const excludeReasoningGlobs = [
      '*tts*',
    ]

    if (!model.name) {
      return {
        tools: true,
        vision: false,
        reasoning: false,
        caching: false,
      }
    }

    // calc
    const modelName = model.name.replace('models/', '')
    // Gemma 3 and earlier expose no native tool tokens, so tool use had to be
    // prompted. Gemma 4 has native function calling and is the only Gemma the
    // Gemini API serves. Carved out by name rather than by lifting the blanket
    // exclusion, so an unrecognised Gemma is still assumed to have no tools.
    const toolCapableGemmaGlobs = [
      'gemma-4*',
    ]

    let tools = (!modelName.includes('gemma') || toolCapableGemmaGlobs.some((m) => minimatch(modelName, m)))
      && !modelName.includes('dialog') && !modelName.includes('tts')
    let vision = visionGlobs.some((m) => minimatch(modelName, m)) && !excludeVisionGlobs.some((m) => minimatch(modelName, m))
    let reasoning = reasoningGlobs.some((m) => minimatch(modelName, m)) && !excludeReasoningGlobs.some((m) => minimatch(modelName, m))

    // latest aliases have all
    if (modelName.endsWith('latest') && !modelName.match(/\d/)) {
      tools = true
      vision = true
      reasoning = true
    }

    // done
    return { tools, vision, reasoning, caching: false }
    
  }

  // models that refuse a thinkingBudget and only take a thinkingLevel
  static levelOnlyThinkingGlobs = [
    'gemini-4-*',
  ]

  /** How this model is told how hard to think: opts, then model meta, then name globs. */
  getThinkingStyle(model: ChatModel, opts?: GoogleCompletionOpts): LlmGoogleThinkingStyle {
    const isStyle = (style: unknown): style is LlmGoogleThinkingStyle => style === 'level' || style === 'budget'
    if (isStyle(opts?.thinkingStyle)) return opts.thinkingStyle
    const metaStyle = (model.meta as { thinking?: unknown } | undefined)?.thinking
    if (isStyle(metaStyle)) return metaStyle
    const modelName = (model.id ?? '').replace('models/', '')
    const globs = (this.constructor as typeof GoogleEngine).levelOnlyThinkingGlobs
    return globs.some((m) => minimatch(modelName, m)) ? 'level' : 'budget'
  }

  /** The thinking config for a level-only model: never a budget, never 'minimal'. */
  protected getLevelThinkingConfig(opts?: GoogleCompletionOpts): ThinkingConfig|undefined {
    // wire values of ThinkingLevel, spelled out so a mocked or trimmed SDK cannot turn them undefined
    const toEnum: Record<LlmGoogleThinkingLevel, ThinkingLevel> = {
      low: 'LOW' as ThinkingLevel,
      medium: 'MEDIUM' as ThinkingLevel,
      high: 'HIGH' as ThinkingLevel,
    }
    if (opts?.reasoning === false) return undefined
    const explicit = typeof opts?.thinkingLevel === 'string' ? toEnum[opts.thinkingLevel.toLowerCase() as LlmGoogleThinkingLevel] : undefined
    if (explicit) return { includeThoughts: true, thinkingLevel: explicit }
    const budget = opts?.thinkingBudget
    // automatic (-1) or nothing asked: the model's default level
    if (typeof budget !== 'number' || budget < 0) return { includeThoughts: true }
    // 0 cannot turn thinking off on these models: it asks for the least
    const thinkingLevel = budget <= 1024 ? toEnum.low : budget <= 8192 ? toEnum.medium : toEnum.high
    return { includeThoughts: true, thinkingLevel }
  }

  isComputerUseModel(model: string): boolean {
    return ['gemini-2.5-computer-use-preview-10-2025'].includes(model)
  }

  async getModels(): Promise<ModelGoogle[]> {

    // need an api key
    if (!this.config.apiKey) {
      return []
    }

    // fpr debugging purposes
    // const actions = new Set<string>()

    // we may have to iterate over multiple pages
    const models: ModelGoogle[] = []
    const pager = await this.client.models.list()
    for await (const model of pager) {
      // model.supportedActions?.forEach((action) => actions.add(action))
      if (!model.name) continue
      //if (model.name.match(/\d\d\d$/)) continue
      if (model.name.includes('tuning')) continue
      if (model.description?.includes('deprecated')) continue
      if (model.description?.includes('discontinued')) continue
      models.push(model as ModelGoogle)
    }

    // debugging
    //console.log(actions)

    // reverse
    models.reverse()

    // done
    return models

  }

  async complete(model: ChatModel, thread: Message[], opts?: LlmCompletionOpts): Promise<LlmResponse> {
    const messages = this.buildGooglePayload(thread, model, opts)
    const instruction = this.getInstructions(model, thread)
    return await this.chat(model, messages, {
      ...opts,
      instruction
    })
  }

  async chat(model: ChatModel, thread: Content[], opts?: GoogleCompletionOpts): Promise<LlmResponse> {

    // save tool calls
    const toolCallInfo: LlmToolCallInfo[] = []
    const startTime = Date.now()

    // call
    logger.log(`[google] prompting model ${model.id}`)
    const response = await this.client.models.generateContent({
      model: model.id,
      contents: thread,
      config: await this.getGenerationConfig(model, opts),
    })

    // check for tool calls
    const toolCalls = response.functionCalls
    if (toolCalls?.length) {

      // results
      const results: FunctionResponse[] = []

      for (const toolCall of toolCalls) {

        // need
        logger.log(`[google] tool call ${toolCall.name} with ${JSON.stringify(toolCall.args)}`)

        // now execute
        let lastUpdate: PluginExecutionResult|undefined = undefined
        for await (const update of this.callTool({ model: model.id, abortSignal: opts?.abortSignal }, toolCall.name!, toolCall.args, opts?.toolExecutionDelegate, opts?.toolExecutionValidation, opts?.toolCallGuard)) {
          if (update.type === 'result') {
            lastUpdate = update
          }
        }

        // process result using helper method
        const { content, canceled } = this.processToolExecutionResult('google', toolCall.name!, toolCall.args, lastUpdate)

        // if canceled/denied, stop processing
        if (canceled) {
          throw new Error('Tool execution was canceled')
        }

        results.push({
          name: toolCall.name!,
          response: toFunctionResponse(content)
        })

        // save tool call info
        toolCallInfo.push({
          name: toolCall.name!,
          params: toolCall.args,
          result: content
        })

      }

      // function call
      thread.push({
        role: 'model',
        parts: response.candidates![0].content!.parts,
      })

      // send. Gemini rejects role 'tool' with a 400: function responses ride in a
      // user turn. See buildGooglePayload and buildToolContents for the same rule.
      thread.push({
        role: 'user',
        parts: results.map((r) => ({ functionResponse: r }) ),
      })

      // apply cooldown before next request
      await this.applyCooldown(startTime)

      // prompt again
      const completion = await this.chat(model, thread, opts)

      // prepend tool call info
      completion.toolCalls = [
        ...toolCallInfo,
        ...completion.toolCalls ?? [],
      ]

      // cumulate usage
      if (opts?.usage && response.usageMetadata && completion.usage) {
        completion.usage.prompt_tokens += (response.usageMetadata.promptTokenCount ?? 0) + (response.usageMetadata.toolUsePromptTokenCount ?? 0)
        completion.usage.completion_tokens += (response.usageMetadata.candidatesTokenCount ?? 0) + (response.usageMetadata.thoughtsTokenCount ?? 0)
        completion.usage.completion_tokens_details!.reasoning_tokens! += response.usageMetadata.thoughtsTokenCount ?? 0
      }

      // done
      return completion

    }

    // thought signature from reasoning if any
    const thoughtSignature = response.candidates?.[0].content?.parts?.find(p => p.thoughtSignature)?.thoughtSignature

    // done
    return {
      type: 'text',
      content: response.text,
      toolCalls: toolCallInfo,
      ...(thoughtSignature ? { thoughtSignature } : {}),
      ...(opts?.usage && response.usageMetadata ? { usage: {
        prompt_tokens: (response.usageMetadata.promptTokenCount ?? 0) + (response.usageMetadata.toolUsePromptTokenCount ?? 0),
        completion_tokens: (response.usageMetadata.candidatesTokenCount ?? 0) + (response.usageMetadata.thoughtsTokenCount ?? 0),
        context_window_tokens: (response.usageMetadata.promptTokenCount ?? 0) + (response.usageMetadata.toolUsePromptTokenCount ?? 0),
        prompt_tokens_details: {
          cached_tokens: response.usageMetadata.cachedContentTokenCount ?? 0,
        },
        completion_tokens_details: {
          reasoning_tokens: response.usageMetadata.thoughtsTokenCount ?? 0,
        },
      } } : {}),
    }
  }

  async stream(model: ChatModel, thread: Message[], opts?: LlmCompletionOpts): Promise<LlmStreamingResponse<GoogleStreamingContext>> {

    // model: switch to vision if needed
    model = this.selectModel(model, thread, opts)

    // add computer plugin if computer use model
    if (this.computerInfo && this.isComputerUseModel(model.id)) {
      const computerPlugin = this.plugins.find((p) => p.getName() === this.computerInfo!.plugin.getName())
      if (!computerPlugin) {
        this.plugins.push(this.computerInfo.plugin)
      }
    }

    // context
    const context: GoogleStreamingContext = {
      model: model,
      thread: this.buildGooglePayload(thread, model, opts),
      opts: {
        ...opts,
        instruction: this.getInstructions(model, thread),
      },
      toolCalls: [],
      toolHistory: [],
      currentRound: 0,
      startTime: 0,
      usage: zeroUsage(),
      requestUsage: zeroUsage()
    }

    // do it
    return {
      stream: await this.doStream(context),
      context: context
    }

  }

  async doStream(context: GoogleStreamingContext): Promise<LlmStream> {

    // reset
    context.toolCalls = []
    context.startTime = Date.now()
    context.requestUsage = zeroUsage()
    context.textContentBlock = undefined

    logger.log(`[google] prompting model ${context.model.id}`)
    const response = await this.client.models.generateContentStream({
      model: context.model.id,
      contents: context.thread,
      config: {
        ...await this.getGenerationConfig(context.model, context.opts),
        ...(context.opts.abortSignal ? { abortSignal: context.opts.abortSignal } : {}),
      },
    })

    // done
    return response

  }

  private supportsInstructions(model: ChatModel): boolean {
    return ['gemini'].some((m) => model.id.includes(m))
  }

  private supportsStructuredOutput(model: ChatModel): boolean {
    return ['gemini'].some((m) => model.id.includes(m))
  }

  private getInstructions(model: ChatModel, thread: Message[]): string|undefined {
    return (this.supportsInstructions(model) && thread.length > 1 && thread[0].role === 'system') ? thread[0].content : undefined  
  }

  private typeToSchemaType(type: string, properties?: any): Type {
    if (type === 'string') return Type.STRING
    if (type === 'number') return Type.NUMBER
    if (type === 'boolean') return Type.BOOLEAN
    if (type === 'array') return Type.ARRAY
    return properties ? Type.OBJECT : Type.STRING
  }

  // Convert a single PluginParameter to Google schema format
  private pluginParamToGoogleSchema(param: PluginParameter): any {
    const type = param.type || (param.items ? 'array' : 'string')
    const prop: any = {
      type: this.typeToSchemaType(type),
      description: param.description,
      ...(param.enum ? { enum: param.enum } : {}),
    }
    if (type === 'array') {
      prop.items = this.convertItemsForGoogle(param.items)
    }
    return prop
  }

  // Convert PluginParameter items to Google format
  private convertItemsForGoogle(items?: { type: string; properties?: any[]; items?: { type: string } }): any {
    if (!items) return { type: this.typeToSchemaType('string') }
    if (!items.properties) {
      if (items.type === 'array' && items.items) {
        return { type: this.typeToSchemaType('array'), items: { type: this.typeToSchemaType(items.items.type || 'string') } }
      }
      return { type: this.typeToSchemaType(items.type) }
    }
    // Convert array of PluginParameter to Record for Google
    const props: Record<string, any> = {}
    for (const prop of items.properties) {
      props[prop.name] = this.pluginParamToGoogleSchema(prop)
    }
    return {
      type: this.typeToSchemaType(items.type, items.properties),
      properties: props,
    }
  }

  protected async getGenerationConfig(model: ChatModel, opts?: GoogleCompletionOpts): Promise<GenerateContentConfig|undefined> {

    const config: GenerateContentConfig = {
      ...(opts?.maxTokens ? { maxOutputTokens: opts?.maxTokens } : {} ),
      ...(opts?.temperature ? { temperature: opts?.temperature } : {} ),
      ...(opts?.top_k ? { topK: opts?.top_k } : {} ),
      ...(opts?.top_p ? { topP: opts?.top_p } : {} ),
    }

    // add instructions
    if (opts?.instruction) {
      config.systemInstruction = opts!.instruction
    }

    // add reasoning
    if (model.capabilities.reasoning && this.getThinkingStyle(model, opts) === 'level') {
      if (typeof opts?.thinkingBudget !== 'undefined' || typeof opts?.thinkingLevel !== 'undefined') {
        const thinkingConfig = this.getLevelThinkingConfig(opts)
        if (thinkingConfig) config.thinkingConfig = thinkingConfig
      }
    } else if (model.capabilities.reasoning && typeof opts?.thinkingBudget !== 'undefined') {
      config.thinkingConfig = {
        includeThoughts: true,
        thinkingBudget: opts.thinkingBudget
      }
    }

    // add structured output
    if (this.supportsStructuredOutput(model) && opts?.structuredOutput) {
      config.responseMimeType = 'application/json'
      config.responseJsonSchema = zodToJsonSchema(opts.structuredOutput.structure)
    }

    // add computer use tool
    if (this.computerInfo && this.isComputerUseModel(model.id)) {
      config.tools = [{
        computerUse: {
          environment: Environment.ENVIRONMENT_BROWSER
        }
      }]
    }

    // add tools
    else if (opts?.tools !== false && model.capabilities.tools) {

      const tools = await this.getAvailableTools(opts?.toolExecutionDelegate);
      if (tools.length) {

        const functionDeclarations: FunctionDeclaration[] = [];

        for (const tool of tools) {

          const googleProps: { [k: string]: Schema } = {};
          const required: string[] = [];

          for (const param of tool.parameters) {
            const type = param.type || (param.items ? 'array' : 'string')
            const prop: any = {
              type: this.typeToSchemaType(type),
              description: param.description,
              ...(param.enum ? { enum: param.enum } : {}),
            }
            if (type === 'array') {
              prop.items = this.convertItemsForGoogle(param.items)
            }
            googleProps[param.name] = prop as Schema
            if (param.required) {
              required.push(param.name)
            }
          }

          functionDeclarations.push({
            name: tool.name,
            description: tool.description,
            ...(tool.parameters.length == 0 ? {} : {
              parameters: {
                type: Type.OBJECT,
                properties: googleProps,
                required: required,
              }
            })
          })
        }

        // done
        config.tools = [{ functionDeclarations: functionDeclarations }]

        // tool call options
        if (opts?.toolChoice?.type === 'none') {
          config.toolConfig = { functionCallingConfig: { mode: FunctionCallingConfigMode.NONE } }
        } else if (opts?.toolChoice?.type === 'required') {
          config.toolConfig = { functionCallingConfig: { mode: FunctionCallingConfigMode.ANY } }
        } else if (opts?.toolChoice?.type === 'tool') {
          config.toolConfig = { functionCallingConfig: {
            mode: FunctionCallingConfigMode.ANY,
            allowedFunctionNames: [ opts.toolChoice.name! ]
          }}
        } else {
          config.toolConfig = { functionCallingConfig: { mode: FunctionCallingConfigMode.AUTO } }
        }

      }
    }

    // done
    return Object.keys(config).length ? config : undefined
  }

  supportsThoughtSignatures(): boolean {
    return true
  }

  requiresFlatTextPayload(model: ChatModel, msg: Message) {
    if (msg.thoughtSignature) return false
    else return super.requiresFlatTextPayload(model, msg)
  }

  public buildGooglePayload(thread: Message[], model: ChatModel, opts?: LlmCompletionOpts): Content[] {
    const supportsInstructions = this.supportsInstructions(model)
    const payload = this.buildPayload(model, thread.filter((m) => supportsInstructions ? m.role !== 'system' : true), opts).map((p) => {
      if (p.role === 'system') p.role = 'user'
      return p
    })

    // A reply holds every tool call of every round it made, followed by one 'tool' payload
    // per call. Gemini (and Vertex strictly) wants a function call turn answered by ONE turn
    // carrying all its function responses, so the reply becomes: its calls, their results,
    // then what it wrote (which it wrote after seeing the results).
    const contents: Content[] = []
    for (let i = 0; i < payload.length; i++) {
      const message = payload[i]
      if (message.role === 'tool') {
        contents.push(this.toolResultsToContent([message as LlmCompletionPayloadTool]))
        continue
      }
      const content = this.messageToContent(message as LlmCompletionPayloadContent)
      const calls = content.parts!.filter((p) => p.functionCall)
      if (!calls.length) {
        contents.push(content)
        continue
      }
      const results: LlmCompletionPayloadTool[] = []
      while (payload[i + 1]?.role === 'tool') results.push(payload[++i] as LlmCompletionPayloadTool)
      contents.push({ role: 'model', parts: calls })
      if (results.length) contents.push(this.toolResultsToContent(results))
      contents.push({ role: 'model', parts: content.parts!.filter((p) => !p.functionCall) })
    }

    return normalizeContents(contents)
  }

  private toolResultsToContent(results: LlmCompletionPayloadTool[]): Content {
    // Gemini's contents accept only 'user' and 'model'; a function response is a
    // user turn carrying a functionResponse part. Sending role 'tool' returns
    // 400 INVALID_ARGUMENT: "Role 'tool' is not supported."
    return {
      role: 'user',
      parts: results.map((result) => {
        let response: unknown = result.content
        try {
          response = JSON.parse(result.content as string)
        } catch {
          // not JSON: sent as text
        }
        return {
          functionResponse: {
            id: result.name,
            name: result.name,
            response: toFunctionResponse(response),
          }
        }
      }),
    }
  }

  private messageToContent(payload: LlmCompletionPayloadContent): Content {

    const content: Content = {
      role: payload.role == 'assistant' ? 'model' : payload.role as 'user' | 'model',
      parts: Array.isArray(payload.content) ? payload.content.map((c) => ({
        text: (c as LLmContentPayloadText).text,
        // ...(c as LlmContentPayload).thoughtSignature ? { thoughtSignature: (c as LlmContentPayload).thoughtSignature } : {},
      })) : [ {
        text: payload.content as string,
      } ],
    }

    if (payload.role === 'assistant' && payload.tool_calls) {
      for (const tc of payload.tool_calls) {
        content.parts!.splice(content.parts!.length-1, 0, {
          functionCall: {
            name: tc.function.name,
            args: parseToolArgs(tc.function.arguments),
          },
          ...(tc.thoughtSignature ? { thoughtSignature: tc.thoughtSignature } : {})
        })
      }
    }

    // add images
    for (const index in payload.images) {
      content.parts!.push({
        inlineData: {
          mimeType: 'image/png',
          data: payload.images[Number(index)],
        }
      })
    }

    // done
    return content

  }

  private addAttachment(parts: Array<string|Part>, attachment: Attachment) {

    // load if no contents
    if (attachment.content === null || attachment.content === undefined) {
      console.warn('[google] attachment contents not available. Skipping attachment.')
    }
  
    // add inline
    if (attachment.isImage()) {
      parts.push({
        inlineData: {
          mimeType: attachment.mimeType,
          data: attachment.content,
        }
      })
    } else if (attachment.isText()) {
      parts.push(attachment.content)
    }

  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async stop(stream: LlmStream) {
    //await stream?.controller?.abort()
  }

  syncToolHistoryToThread(context: GoogleStreamingContext): void {
    // sync mutations from toolHistory back to content
    // Google content format: { role: 'user', parts: [{ functionResponse: { id, name, response } }] }
    // Matched on the functionResponse part rather than the role: function responses
    // share the 'user' role with real user turns, and only they carry this part.
    // Google doesn't manage IDs, so we match by name and index position
    let historyIndex = 0
    for (const msg of context.thread) {
      if (!msg.parts) continue
      for (const part of msg.parts as Part[]) {
        const functionResponse = part.functionResponse as FunctionResponse | undefined
        if (functionResponse && historyIndex < context.toolHistory.length) {
          const entry = context.toolHistory[historyIndex]
          if (functionResponse.name === entry.name && functionResponse.response !== entry.result) {
            functionResponse.response = (typeof entry.result === 'string') ? { result: entry.result } : entry.result
          }
          historyIndex++
        }
      }
    }
  }

  async *processNativeChunk(chunk: GenerateContentResponse, context: GoogleStreamingContext): AsyncGenerator<LlmChunk> {

    // debug
    //logger.log('[google] chunk', JSON.stringify(chunk))

    // usage
    if (chunk.usageMetadata) {
      context.requestUsage.prompt_tokens = (chunk.usageMetadata.promptTokenCount ?? 0) + (chunk.usageMetadata.toolUsePromptTokenCount ?? 0)
      context.requestUsage.context_window_tokens = (chunk.usageMetadata.promptTokenCount ?? 0) + (chunk.usageMetadata.toolUsePromptTokenCount ?? 0)
      context.requestUsage.prompt_tokens_details!.cached_tokens = chunk.usageMetadata.cachedContentTokenCount ?? 0
      context.requestUsage.completion_tokens = (chunk.usageMetadata.candidatesTokenCount ?? 0) + (chunk.usageMetadata.thoughtsTokenCount ?? 0)
      context.requestUsage.completion_tokens_details!.reasoning_tokens! = chunk.usageMetadata.thoughtsTokenCount ?? 0
    }

    // tool calls - normalize and process (Google sends complete args, no deltas)
    const toolParts: Part[] = chunk.candidates?.[0].content?.parts?.filter(p => p.functionCall?.name) || []
    for (const part of toolParts) {
      const tc = part.functionCall!
      yield* this.processToolCallChunk({
        type: 'start',
        id: tc.id || crypto.randomUUID(),
        name: tc.name!,
        args: JSON.stringify(tc.args),
        metadata: {
          thoughtSignature: part.thoughtSignature,
        }
      }, context)
    }

    // check for finish reason and accumulated tool calls
    const done = !!chunk.candidates?.[0].finishReason
    if (done && context.toolCalls?.length) {

      // clear force tool call to avoid infinite loop
      if (context.opts.toolChoice?.type === 'tool') {
        delete context.opts.toolChoice
      }

      // increment round for next iteration
      context.currentRound++

      // add usage before continuing
      if (context.opts.usage) {
        context.usage = addUsages(context.usage, context.requestUsage)
        context.requestUsage = zeroUsage()
      }

      // execute tool calls using base class method
      yield* this.executeToolCallsBatched(context.toolCalls, context, {
        formatBatchForThread: (completed) => {
          // build assistant parts: text first (if present), then function calls
          const assistantParts: Part[] = []

          // add accumulated text content if present
          if (context.textContentBlock) {
            assistantParts.push({ text: context.textContentBlock })
          }

          // add function calls
          for (const { tc, args } of completed) {
            assistantParts.push({
              functionCall: {
                name: tc.function,
                args: args,
              },
              ...(tc.thoughtSignature ? { thoughtSignature: tc.thoughtSignature } : {}),
            })
          }

          // assistant message with all content
          const assistantContent: Content = {
            role: 'model',
            parts: assistantParts,
          }

          // tool message with all function responses. Role is 'user', not 'tool':
          // Gemini rejects 'tool' outright.
          const toolContent: Content = {
            role: 'user',
            parts: completed.map(({ tc, result }) => ({
              functionResponse: {
                id: tc.function,
                name: tc.function,
                response: toFunctionResponse(result)
              } as FunctionResponse
            })),
          }

          return [assistantContent, toolContent]
        },
        createNewStream: async () => this.doStream(context)
      })

      // done
      return

    }

    // iterate on candidates (content and reasoning)
    for (const candidate of chunk.candidates || []) {
      for (const part of candidate.content?.parts || []) {
        const text = part.text || ''
        // accumulate text content (not thought/reasoning)
        if (!part.thought && text) {
          context.textContentBlock = (context.textContentBlock || '') + text
        }
        yield {
          type: part.thought ? 'reasoning' : 'content',
          text: text,
          ...(part.thoughtSignature ? { thoughtSignature: part.thoughtSignature } : {}),
          done: done
        }
      }
    }

    // usage
    if (done && context.opts.usage) {
      context.usage = addUsages(context.usage, context.requestUsage)
      yield { type: 'usage', usage: context.usage }
    }
  }
   
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  addImageToPayload(model: ChatModel, attachment: Attachment, payload: LlmCompletionPayloadContent, opts?: LlmCompletionOpts) {
    if (!payload.images) payload.images = []
    payload.images.push(attachment!.content)
  }

}

/** Gemini's functionResponse.response is a Struct: anything else (text, a list, a number) is wrapped. */
export const toFunctionResponse = (response: unknown): Record<string, unknown> =>
  typeof response === 'object' && response !== null && !Array.isArray(response)
    ? response as Record<string, unknown>
    : { result: response }

/** A tool call stored without args (interrupted before they arrived) is sent with none. */
const parseToolArgs = (args: unknown): Record<string, unknown> => {
  if (typeof args !== 'string' || args.trim() === '') return {}
  try {
    const parsed = JSON.parse(args)
    return typeof parsed === 'object' && parsed !== null && !Array.isArray(parsed) ? parsed : {}
  } catch {
    return {}
  }
}

const isEmptyText = (part: Part): boolean =>
  'text' in part && !part.text && Object.keys(part).every((key) => key === 'text' || key === 'thoughtSignature')

/** Drops empty text parts and turns left empty, then merges consecutive turns of the same role. */
const normalizeContents = (contents: Content[]): Content[] => {
  const normalized: Content[] = []
  for (const content of contents) {
    const parts = (content.parts ?? []).filter((part) => !isEmptyText(part))
    if (!parts.length) continue
    const previous = normalized[normalized.length - 1]
    if (previous && previous.role === content.role) {
      previous.parts!.push(...parts)
    } else {
      normalized.push({ ...content, parts })
    }
  }
  return normalized
}
