import logging
logger = logging.getLogger('django')

import os
# 禁用 browser-use 遥测
os.environ['ANONYMIZED_TELEMETRY'] = 'false'

import asyncio
import functools
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

# ============================================================================
# PART 1: Common Patches (Pydantic, ActionModel, TokenCost, Basic Connection)
# ============================================================================

# Patch ChatOpenAI to allow setting attributes (required for browser-use token counting)
try:
    from pydantic import ConfigDict
    if hasattr(ChatOpenAI, 'model_config'):
        if isinstance(ChatOpenAI.model_config, dict):
            ChatOpenAI.model_config['extra'] = 'allow'
        else:
            ChatOpenAI.model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
    else:
        ChatOpenAI.model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)
except ImportError:
    if hasattr(ChatOpenAI, 'model_config'):
        ChatOpenAI.model_config['extra'] = 'allow'

# 修改 ActionModel 配置以允许额外字段
try:
    from browser_use.tools.registry.views import ActionModel
    from pydantic import ConfigDict
    ActionModel.model_config = ConfigDict(arbitrary_types_allowed=True, extra='allow')
    logger.info("[OK] Modified ActionModel.model_config to allow extra fields")
except Exception as e:
        logger.warning(f"[WARNING] Failed to modify ActionModel config: {e}")

# Patch Agent.get_model_output 方法
try:
    from browser_use.agent.service import Agent
    from browser_use.agent.message_manager.service import AgentOutput
    import json as json_module

    _original_get_model_output = Agent.get_model_output

    async def _patched_get_model_output(self, input_messages):
        """修补后的 get_model_output，直接从 response.content 解析 JSON"""
        # logger.info("[TOOL] _patched_get_model_output called")

        if hasattr(self, '_task_was_done') and self._task_was_done:
            logger.info("[OPT] Task was marked as done, stopping LLM interaction")
            raise KeyboardInterrupt("Task finished")

        kwargs = {'output_format': self.AgentOutput}

        # Add retry logic for LLM invocation with timeout
        max_retries = 2  # 重试次数为2次
        last_exception = None
        response = None
        for attempt in range(max_retries):
            try:
                # 添加超时控制，设置为30秒
                response = await asyncio.wait_for(
                    self.llm.ainvoke(input_messages, **kwargs),
                    timeout=30.0  # 超时时间30秒
                )
                break
            except asyncio.TimeoutError as te:
                last_exception = te
                logger.warning(f"[WARNING] LLM invocation timed out (attempt {attempt+1}/{max_retries}): {te}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # 重试间隔0.5秒
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # 检查是否是账户余额不足错误
                if 'balance' in error_str.lower() or 'insufficient' in error_str.lower() or '30001' in error_str:
                    logger.error(f"[ERROR] API账户余额不足: {error_str}")
                    error_msg = "API账户余额不足，请充值后重试"
                    raise ValueError(error_msg) from e
                
                # 检查是否是403错误
                if '403' in error_str or 'forbidden' in error_str.lower():
                    logger.error(f"[ERROR] API访问被拒绝(403): {error_str}")
                    if 'balance' in error_str.lower() or 'insufficient' in error_str.lower():
                        error_msg = "API账户余额不足，请充值后重试"
                    else:
                        error_msg = f"API访问被拒绝: {error_str}"
                    raise ValueError(error_msg) from e
                
                logger.warning(f"[WARNING] LLM invocation failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # 重试间隔0.5秒
        else:
            logger.error(f"[ERROR] LLM invocation failed after {max_retries} attempts.")
            # 检查最后的异常是否是余额不足
            if last_exception:
                error_str = str(last_exception)
                if 'balance' in error_str.lower() or 'insufficient' in error_str.lower() or '30001' in error_str:
                    error_msg = "API账户余额不足，请充值后重试"
                    raise ValueError(error_msg) from last_exception
            raise last_exception

        # 检查响应是否为空或无效
        if not response or not hasattr(response, 'content'):
            error_msg = "LLM returned invalid response (no content attribute)"
            logger.error(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        # 检查content是否为空字符串
        content = response.content
        if not content or not isinstance(content, str) or not content.strip():
            error_msg = "LLM returned empty content - possible API error or timeout"
            logger.error(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        try:
            if hasattr(response, 'content') and isinstance(response.content, str):
                content_dict = json_module.loads(response.content)
                
                # 规范化 action 字典
                if 'action' in content_dict:
                    normalized_actions = []
                    for action_dict in content_dict['action']:
                        normalized_action = {}
                        for action_name, action_params in action_dict.items():
                            # 自动修复: 将 int 参数转换为 index 字典
                            if isinstance(action_params, int):
                                normalized_action[action_name] = {'index': action_params}
                            # 自动修复: switch_tab 的 tab_id 字符串参数
                            elif action_name == 'switch_tab' and isinstance(action_params, str) and not isinstance(action_params, dict):
                                normalized_action[action_name] = {'tab_id': action_params}
                            elif isinstance(action_params, dict):
                                normalized_params = {}
                                for k, v in action_params.items():
                                    if k == 'element_index':
                                        normalized_params['index'] = v
                                    else:
                                        normalized_params[k] = v
                                normalized_action[action_name] = normalized_params
                            else:
                                normalized_action[action_name] = action_params
                        normalized_actions.append(normalized_action)
                    content_dict['action'] = normalized_actions
                
                parsed = AgentOutput.model_construct(
                    thinking=content_dict.get('thinking'),
                    evaluation_previous_goal=content_dict.get('evaluation_previous_goal'),
                    memory=content_dict.get('memory'),
                    next_goal=content_dict.get('next_goal'),
                    action=[]
                )
                
                class _ActionWrapper:
                    def __init__(self, action_dict):
                        self._action_dict = action_dict
                    def model_dump(self, **kwargs):
                        return self._action_dict
                    def get_index(self):
                        for action_params in self._action_dict.values():
                            if isinstance(action_params, dict) and 'index' in action_params:
                                return action_params['index']
                        return None

                action_list = []
                for action_dict in content_dict.get('action', []):
                    action_list.append(_ActionWrapper(action_dict))
                
                object.__setattr__(parsed, 'action', action_list)
                
                if len(parsed.action) > self.settings.max_actions_per_step:
                    parsed.action = parsed.action[:self.settings.max_actions_per_step]

                return parsed
        except Exception as e:
            # If our complex normalization fails, fall back to the original method
            logger.warning(f"[WARNING] Custom output normalization failed, falling back: {e}")
            return await _original_get_model_output(self, input_messages)
    
    Agent.get_model_output = _patched_get_model_output
    logger.info("[OK] Successfully patched Agent.get_model_output")
except Exception as e:
    logger.error(f"[ERROR] Failed to patch Agent.get_model_output: {e}")

# Patch TokenCost
try:
    from browser_use.tokens.service import TokenCost
    from langchain_core.messages import HumanMessage, SystemMessage as LangChainSystemMessage, AIMessage

    def _patched_register_llm(self, llm):
        """修补后的 register_llm，修复 langchain 兼容性"""
        instance_id = str(id(llm))
        if instance_id in self.registered_llms:
            return llm
        
        self.registered_llms[instance_id] = llm
        _original_ainvoke = llm.ainvoke
        _token_service = self
        
        async def _fixed_tracked_ainvoke(messages, output_format=None, **kwargs):
            # Sanitize message contents
            def _content_to_str(content):
                if isinstance(content, str): return content
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, str): parts.append(item)
                        elif isinstance(item, dict):
                            if 'text' in item: parts.append(str(item['text']))
                            elif 'image' in item or 'image_url' in item: parts.append("[image]")
                        else: parts.append(str(item))
                    return "\n".join(parts)
                if isinstance(content, dict):
                    if 'text' in content: return str(content['text'])
                    if 'content' in content: return str(content['content'])
                    if 'image' in content or 'image_url' in content: return "[image]"
                return str(content)

            def _sanitize_message(msg):
                msg_type_name = type(msg).__name__
                content = getattr(msg, 'content', msg)
                content_str = _content_to_str(content)
                if msg_type_name == 'SystemMessage': return LangChainSystemMessage(content=content_str)
                if msg_type_name in ('HumanMessage', 'UserMessage'): return HumanMessage(content=content_str)
                if msg_type_name == 'AIMessage': return AIMessage(content=content_str)
                if isinstance(msg, (HumanMessage, LangChainSystemMessage, AIMessage)): return type(msg)(content=content_str)
                return HumanMessage(content=str(content_str))

            sanitized_messages = [_sanitize_message(m) for m in messages]
            
            output_format = kwargs.pop('output_format', None)
            if output_format:
                kwargs['response_format'] = {"type": "json_object"}
                
            # Add retry logic for LLM invocation
            max_retries = 2  # 重试次数为2次
            last_exception = None
            for attempt in range(max_retries):
                try:
                    result = await _original_ainvoke(sanitized_messages, **kwargs)
                    break
                except Exception as e:
                    last_exception = e
                    error_str = str(e)
                    
                    # 检查是否是账户余额不足错误
                    if 'balance' in error_str.lower() or 'insufficient' in error_str.lower() or '30001' in error_str:
                        logger.error(f"[ERROR] API账户余额不足: {error_str}")
                        error_msg = "API账户余额不足，请充值后重试"
                        raise ValueError(error_msg) from e
                    
                    # 检查是否是403错误
                    if '403' in error_str or 'forbidden' in error_str.lower():
                        logger.error(f"[ERROR] API访问被拒绝(403): {error_str}")
                        if 'balance' in error_str.lower() or 'insufficient' in error_str.lower():
                            error_msg = "API账户余额不足，请充值后重试"
                        else:
                            error_msg = f"API访问被拒绝: {error_str}"
                        raise ValueError(error_msg) from e
                    
                    if "response_format" in error_str:
                        kwargs.pop('response_format', None)
                        # retry immediately without response_format
                        continue

                    logger.warning(f"[WARNING] LLM ainvoke failed (attempt {attempt+1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)  # 等待0.5秒
            else:
                logger.error(f"[ERROR] LLM ainvoke failed after {max_retries} attempts.")
                # 检查最后的异常是否是余额不足
                if last_exception:
                    error_str = str(last_exception)
                    if 'balance' in error_str.lower() or 'insufficient' in error_str.lower() or '30001' in error_str:
                        error_msg = "API账户余额不足，请充值后重试"
                        raise ValueError(error_msg) from last_exception
                raise last_exception

            # Enhance response parsing
            import json as json_module
            clean_content = result.content.strip() if hasattr(result, 'content') else str(result).strip()
            
            # Remove Markdown
            if '```' in clean_content:
                match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', clean_content, re.DOTALL)
                if match: clean_content = match.group(1).strip()
                else: clean_content = re.sub(r'```[a-z]*', '', clean_content).replace('```', '').strip()

            parsed_data = None
            try:
                parsed_data = json_module.loads(clean_content)
            except:
                try:
                    match = re.search(r'(\{.*\})', clean_content, re.DOTALL)
                    if match: parsed_data = json_module.loads(match.group(1))
                except:
                    pass
            
            # Wrapper classes
            class _ActionWrapper:
                def __init__(self, action_dict):
                    self._dict = {}
                    for k, v in action_dict.items():
                        if isinstance(v, dict):
                            norm = {}
                            for subk, subv in v.items():
                                if subk == 'element_index': norm['index'] = subv
                                else: norm[subk] = subv
                            self._dict[k] = norm
                        else: self._dict[k] = v
                    for k, v in self._dict.items(): setattr(self, k, v)
                def model_dump(self, **kwargs): return self._dict
                def get_index(self):
                    for v in self._dict.values():
                        if isinstance(v, dict) and 'index' in v: return v['index']
                    return None
            
            # Construct AgentOutput manually
            agent_output = None
            if parsed_data and 'action' in parsed_data:
                # Normalize actions
                normalized_actions = []
                for action_dict in parsed_data['action']:
                    normalized_action = {}
                    for action_name, action_params in action_dict.items():
                        if isinstance(action_params, dict):
                            normalized_params = {}
                            for k, v in action_params.items():
                                if k == 'element_index': normalized_params['index'] = v
                                else: normalized_params[k] = v
                            normalized_action[action_name] = normalized_params
                        else:
                            normalized_action[action_name] = action_params
                    normalized_actions.append(normalized_action)
                parsed_data['action'] = normalized_actions

                try:
                    from browser_use.agent.message_manager.service import AgentOutput
                    agent_output = AgentOutput.model_construct(
                        thinking=parsed_data.get('thinking'),
                        evaluation_previous_goal=parsed_data.get('evaluation_previous_goal'),
                        memory=parsed_data.get('memory'),
                        next_goal=parsed_data.get('next_goal'),
                        action=[]
                    )
                    action_list = []
                    for action_dict in parsed_data.get('action', []):
                        action_list.append(_ActionWrapper(action_dict))
                    object.__setattr__(agent_output, 'action', action_list)
                except Exception as e:
                    logger.error(f"[ERROR] Failed to create AgentOutput: {e}")

            class _ResponseWrapper:
                def __init__(self, orig, completion_obj):
                    self._orig = orig
                    self.content = getattr(orig, 'content', '')
                    self.response_metadata = getattr(orig, 'response_metadata', {})
                    self.completion = completion_obj
                    usage = getattr(orig, 'usage', None) or (orig.response_metadata.get('token_usage') if hasattr(orig, 'response_metadata') else None)
                    if not usage: usage = {}
                    # Fix usage
                    usage = dict(usage) if hasattr(usage, '__dict__') else usage
                    usage.setdefault('prompt_tokens', 0)
                    usage.setdefault('completion_tokens', 0)
                    usage.setdefault('total_tokens', 0)
                    self.usage = usage
                def __getattr__(self, name): return getattr(self._orig, name)

            wrapped = _ResponseWrapper(result, agent_output)
            if hasattr(wrapped, 'usage') and wrapped.usage:
                try: _token_service.add_usage(llm.model, wrapped.usage)
                except: pass
            
            return wrapped
        
        setattr(llm, 'ainvoke', _fixed_tracked_ainvoke)
        return llm
    
    TokenCost.register_llm = _patched_register_llm
    logger.info("[OK] Successfully patched TokenCost.register_llm")
except Exception as e:
    logger.error(f"[ERROR] Failed to patch TokenCost: {e}")

# Patch BrowserSession.connect (Windows CDP fix)
try:
    from browser_use.browser.session import BrowserSession
    import httpx
    
    _original_connect = BrowserSession.connect
    
    async def _patched_connect(self, cdp_url=None):
        # 如果提供了 cdp_url，需要确保它是 WebSocket URL 格式
        if cdp_url:
            # 检查并转换 HTTP URL 为 WebSocket URL
            if cdp_url.startswith('http://'):
                # 从 HTTP URL 提取端口，然后从 CDP endpoint 获取 WebSocket URL
                try:
                    import re
                    port_match = re.search(r':(\d+)', cdp_url)
                    if port_match:
                        port = int(port_match.group(1))
                        # 使用 127.0.0.1 而不是 localhost，避免 VPN 路由问题
                        cdp_endpoint = f"http://127.0.0.1:{port}/json/version"
                        
                        # 尝试从 endpoint 获取 WebSocket URL
                        async with httpx.AsyncClient(timeout=10.0) as client:
                            for retry in range(3):
                                try:
                                    response = await client.get(cdp_endpoint)
                                    if response.status_code == 200 and response.text:
                                        version_info = response.json()
                                        if 'webSocketDebuggerUrl' in version_info:
                                            cdp_url = version_info['webSocketDebuggerUrl']
                                            logger.info(f"[INFO] Converted HTTP URL to WebSocket URL: {cdp_url}")
                                            break
                                except Exception as e:
                                    if retry < 2:
                                        await asyncio.sleep(1.0)
                                        continue
                                    logger.warning(f"[WARNING] Failed to get WebSocket URL from endpoint: {e}")
                    else:
                        # 如果无法提取端口，尝试直接转换
                        cdp_url = cdp_url.replace('http://', 'ws://').replace('https://', 'wss://')
                        logger.info(f"[INFO] Converted HTTP URL to WebSocket URL: {cdp_url}")
                except Exception as e:
                    logger.warning(f"[WARNING] Failed to convert CDP URL format: {e}, using as-is")
            
            logger.info(f"[INFO] Using provided CDP URL: {cdp_url}")
            try:
                return await _original_connect(self, cdp_url=cdp_url)
            except Exception as e:
                error_str = str(e)
                # 如果是 JSON 解析错误，可能是 URL 格式问题，尝试重新获取
                if 'JSONDecodeError' in error_str or 'Expecting value' in error_str:
                    logger.warning(f"[WARNING] CDP URL may be invalid, will try to re-fetch: {e}")
                    # 清除 cdp_url，让下面的逻辑重新获取
                    cdp_url = None
                else:
                    raise
        
        browser_profile = getattr(self, 'browser_profile', None)
        if not browser_profile:
            logger.warning("[WARNING] Browser profile not found, using original connect")
            return await _original_connect(self, cdp_url=cdp_url)
            
        # 如果 browser_profile 已经有 cdp_url，先尝试使用，如果失败则重新获取
        if hasattr(browser_profile, 'cdp_url') and browser_profile.cdp_url:
            logger.info(f"[INFO] Using existing CDP URL from browser_profile: {browser_profile.cdp_url}")
            try:
                # 等待一小段时间确保浏览器完全启动
                await asyncio.sleep(0.5)
                return await _original_connect(self, cdp_url=browser_profile.cdp_url)
            except Exception as e:
                error_str = str(e)
                # 如果 CDP URL 无效，清除它并重新获取
                if 'JSONDecodeError' in error_str or 'Expecting value' in error_str or 'connection refused' in error_str.lower():
                    logger.warning(f"[WARNING] Existing CDP URL is invalid, clearing and re-fetching: {e}")
                    browser_profile.cdp_url = None
                    # 继续执行下面的逻辑重新获取 CDP URL
                else:
                    raise
        
        port = 9222
        if hasattr(browser_profile, 'extra_chromium_args'):
            for arg in browser_profile.extra_chromium_args:
                if '--remote-debugging-port=' in str(arg):
                    try: 
                        port = int(arg.split('=')[1])
                        break
                    except: 
                        pass
        if hasattr(browser_profile, 'remote_debugging_port'):
            port = browser_profile.remote_debugging_port
            
        # 使用 127.0.0.1 而不是 localhost，避免 VPN 路由问题
        cdp_endpoint = f"http://127.0.0.1:{port}/json/version"
        
        # 增加重试次数和等待时间，给浏览器更多时间启动
        for attempt in range(10):
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    response = await client.get(cdp_endpoint)
                    
                    # 检查响应状态码
                    if response.status_code != 200:
                        # 502 错误通常是 VPN 路由 localhost 流量导致的
                        if response.status_code == 502:
                            logger.warning(f"CDP endpoint returned status 502 (Bad Gateway), attempt {attempt + 1}/10")
                            logger.warning("[WARNING] This may be caused by VPN routing localhost traffic. Ensure VPN is not intercepting local connections.")
                            # 对于502错误，尝试使用127.0.0.1（如果之前用的是localhost）
                            if 'localhost' in cdp_endpoint:
                                cdp_endpoint = cdp_endpoint.replace('localhost', '127.0.0.1')
                                logger.info(f"[INFO] Retrying with 127.0.0.1 instead of localhost")
                        else:
                            logger.warning(f"CDP endpoint returned status {response.status_code}, attempt {attempt + 1}/10")
                        if attempt < 9:
                            await asyncio.sleep(2.0)  # 增加等待时间
                        continue
                    
                    # 检查响应内容是否为空
                    if not response.text or not response.text.strip():
                        logger.warning(f"CDP endpoint returned empty response, attempt {attempt + 1}/10")
                        if attempt < 9:
                            await asyncio.sleep(2.0)
                        continue
                    
                    # 验证是否为有效的 JSON
                    try:
                        version_info = response.json()
                    except Exception as json_error:
                        logger.error(f"Failed to parse CDP response as JSON (attempt {attempt + 1}/10): {json_error}")
                        logger.debug(f"Response content: {response.text[:200]}")  # 只记录前200个字符
                        if attempt < 9:
                            await asyncio.sleep(2.0)
                        continue
                    
                    # 检查是否包含必需的字段
                    if 'webSocketDebuggerUrl' not in version_info:
                        logger.error(f"CDP response missing 'webSocketDebuggerUrl' field (attempt {attempt + 1}/10)")
                        logger.debug(f"Response keys: {version_info.keys()}")
                        if attempt < 9:
                            await asyncio.sleep(2.0)
                        continue
                    
                    # 成功获取 CDP URL
                    cdp_ws_url = version_info.get('webSocketDebuggerUrl')
                    if not cdp_ws_url:
                        logger.error(f"[ERROR] CDP response missing 'webSocketDebuggerUrl' field")
                        logger.debug(f"Response content: {version_info}")
                        if attempt < 9:
                            await asyncio.sleep(2.0)
                            continue
                        else:
                            raise ValueError("CDP response missing 'webSocketDebuggerUrl' field")
                    
                    browser_profile.cdp_url = cdp_ws_url
                    logger.info(f"[INFO] Successfully retrieved CDP URL: {browser_profile.cdp_url}")
                    
                    # 在连接前，先等待一小段时间确保浏览器完全启动
                    # 这样可以避免浏览器还未完全启动就尝试连接导致的 JSON 解析错误
                    if attempt < 3:  # 前几次尝试等待更长时间
                        await asyncio.sleep(1.0)
                    else:
                        await asyncio.sleep(0.5)
                    
                    # 执行连接（使用更长的超时时间）
                    try:
                        result = await asyncio.wait_for(
                            _original_connect(self, cdp_url=browser_profile.cdp_url),
                            timeout=30.0  # 增加连接超时到30秒
                        )
                        # 连接成功，等待 CDP 客户端完全初始化
                        await asyncio.sleep(0.5)
                        logger.info(f"[INFO] CDP connection established successfully")
                        return result
                    except asyncio.TimeoutError:
                        logger.error("[ERROR] Browser CDP connection timeout after 30 seconds")
                        # 清除无效的 CDP URL
                        browser_profile.cdp_url = None
                        if attempt < 9:
                            await asyncio.sleep(3.0)
                            continue
                        raise
                    except Exception as connect_error:
                        # 捕获连接过程中的任何错误，包括 JSON 解析错误
                        error_str = str(connect_error)
                        if 'JSONDecodeError' in error_str or 'Expecting value' in error_str:
                            logger.error(f"[ERROR] CDP connection failed due to JSON parsing error: {connect_error}")
                            logger.error(f"[ERROR] This may indicate the browser CDP endpoint returned invalid data")
                            # 清除无效的 CDP URL
                            browser_profile.cdp_url = None
                            # 尝试重试
                            if attempt < 9:
                                logger.warning(f"[WARNING] Retrying CDP connection (attempt {attempt + 2}/10)")
                                await asyncio.sleep(3.0)  # 等待更长时间
                                continue
                        elif 'connection refused' in error_str.lower() or 'cannot connect' in error_str.lower():
                            logger.warning(f"[WARNING] CDP connection refused, retrying (attempt {attempt + 1}/10)")
                            browser_profile.cdp_url = None
                            if attempt < 9:
                                await asyncio.sleep(2.0)
                                continue
                        raise
                    
            except httpx.TimeoutException:
                logger.warning(f"CDP endpoint timeout (attempt {attempt + 1}/10), browser may not be ready yet")
                if attempt < 9:
                    await asyncio.sleep(2.0)
            except httpx.ConnectError:
                logger.warning(f"Cannot connect to CDP endpoint at {cdp_endpoint} (attempt {attempt + 1}/10), browser may not be ready yet")
                if attempt < 9:
                    await asyncio.sleep(2.0)
            except Exception as e:
                logger.error(f"Unexpected error connecting to CDP endpoint (attempt {attempt + 1}/10): {e}")
                if attempt < 9:
                    await asyncio.sleep(2.0)
        
        # 所有重试都失败了，记录错误并尝试使用原始连接
        logger.error(f"Failed to connect to CDP endpoint after 10 attempts: {cdp_endpoint}")
        logger.error("Browser may not have started properly or CDP endpoint is not available")
        return await _original_connect(self, cdp_url=cdp_url)
    
    BrowserSession.connect = _patched_connect
    logger.info("[OK] Successfully patched BrowserSession.connect")
except Exception as e:
    logger.error(f"[ERROR] Failed to patch BrowserSession.connect: {e}")

# Patch ClickElementAction parameters
try:
    from browser_use.tools.views import ClickElementAction
    _original_click_init = ClickElementAction.__init__
    def _patched_click_init(self, **kwargs):
        fixed_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, int) and key not in ['index']:
                fixed_kwargs['index'] = value
            else:
                fixed_kwargs[key] = value
        if len(kwargs) == 1:
            key, value = list(kwargs.items())[0]
            if isinstance(value, int) and key != 'index':
                fixed_kwargs = {'index': value}
        try:
            return _original_click_init(self, **fixed_kwargs)
        except TypeError:
            if fixed_kwargs and isinstance(list(fixed_kwargs.values())[0], int):
                return _original_click_init(self, **{'index': list(fixed_kwargs.values())[0]})
            raise
    ClickElementAction.__init__ = _patched_click_init
except Exception:
    pass

# Patch ToolRegistry
try:
    from browser_use.tools.registry.service import Registry as ToolRegistry
    # Force patch Registry class
    _original_execute_action = ToolRegistry.execute_action
    
    async def _patched_execute_action(self, action_name: str, params: dict, **kwargs):
        # 自动映射 switch_tab -> switch (强制映射)
        if action_name == 'switch_tab':
            logger.info(f"[OPT] Force aliasing: switch_tab -> switch")
            action_name = 'switch'
        
        if isinstance(params, int):
            params = {'index': params}
        elif not isinstance(params, dict) and params is not None:
            # 针对 switch_tab 可能是纯字符串的情况
            if action_name in ['switch_tab', 'switch']:
                params = {'tab_id': params}
            else:
                params = {'value': params} if params else {}
        
        # 针对点击增加延迟，确保 UI 更新 (如弹窗弹出、下拉框展开)
        if action_name in ['click_element', 'click']:
            result = await _original_execute_action(self, action_name, params, **kwargs)
            # 增加延迟到 1.5s，并强制在点击后等待浏览器渲染
            # 尤其是对于 element-plus 等 UI 框架，下拉列表渲染需要时间
            await asyncio.sleep(1.5)
            return result
        
        # 针对输入操作增加延迟，确保 GIF 录制能捕获到输入过程
        # 这对于用户名、密码等关键输入步骤的录制特别重要
        if action_name in ['input_text', 'fill', 'type', 'input']:
            result = await _original_execute_action(self, action_name, params, **kwargs)
            # 输入后等待 1.0 秒，确保：
            # 1. 输入值已正确填充到输入框
            # 2. 浏览器的截图能够捕获到输入后的状态
            # 3. GIF 录制能够包含输入操作的完整过程
            # 这对于密码输入等敏感操作的可视化回放非常重要
            await asyncio.sleep(1.0)
            # 额外的短暂等待，确保表单验证等异步操作完成
            await asyncio.sleep(0.3)
            return result
            
        return await _original_execute_action(self, action_name, params, **kwargs)
        
    ToolRegistry.execute_action = _patched_execute_action
    logger.info("[OK] Successfully patched ToolRegistry.execute_action with alias support")
except Exception as e:
    logger.error(f"[ERROR] Failed to patch ToolRegistry: {e}")

# Patch ScreenshotWatchdog GLOBALLY to fix timeouts
try:
    from browser_use.browser.watchdogs.screenshot_watchdog import ScreenshotWatchdog
    
    _original_on_screenshot_event = ScreenshotWatchdog.on_ScreenshotEvent
    
    # Check if already patched to avoid double patching
    if not getattr(_original_on_screenshot_event, '_is_patched_global', False):
        async def on_ScreenshotEvent(self, event):
            """
            Patched screenshot event handler with increased timeout and optimized parameters.
            """
            try:
                # Try original method first with strict timeout
                result = await asyncio.wait_for(
                    _original_on_screenshot_event(self, event),
                    timeout=3.0 # Reduced for fail-fast
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(f"DEBUG: Watchdog timeout (3s), trying optimized approach...")
                try:
                    # Get CDP session
                    cdp_session = await self.browser_session.get_or_create_cdp_session(target_id=None)
                    if not cdp_session: raise Exception("Failed to get CDP session")
                    
                    params = {'format': 'png', 'quality': 50, 'from_surface': True, 'capture_beyond_viewport': False}
                    
                    # One quick retry
                    result = await asyncio.wait_for(
                        cdp_session.cdp_client.send.Page.captureScreenshot(params=params, session_id=cdp_session.session_id),
                        timeout=3.0
                    )
                    return result
                
                except Exception as ex:
                    # In Text Mode especially, we don't want to die on screenshot
                    logger.warning(f"DEBUG: Screenshot failed optimized, returning placeholder: {ex}")
                    import base64
                    # 1x1 transparent pixel
                    placeholder = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                    return {'data': placeholder}
            except Exception as e:
                logger.error(f"DEBUG: Screenshot unexpected error: {e}")
                import base64
                placeholder = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==')
                return {'data': placeholder}
        
        on_ScreenshotEvent._is_patched_global = True
        ScreenshotWatchdog.on_ScreenshotEvent = on_ScreenshotEvent
        logger.info("[OK] Applied Global ScreenshotWatchdog Patch")

    # Patch DOMWatchdog
    from browser_use.browser.watchdogs.dom_watchdog import DOMWatchdog
    
    _original_capture_clean_screenshot = DOMWatchdog._capture_clean_screenshot
    
    if not getattr(_original_capture_clean_screenshot, '_is_patched_global', False):
        async def _capture_clean_screenshot(self):
            try:
                # Very short timeout for DOM clean screenshot checks
                return await asyncio.wait_for(_original_capture_clean_screenshot(self), timeout=3.0)
            except Exception as e:
                logger.warning(f"DEBUG: Clean screenshot failed/timed out: {e}, continuing...")
                return None
                
        _capture_clean_screenshot._is_patched_global = True
        DOMWatchdog._capture_clean_screenshot = _capture_clean_screenshot
        logger.info("[OK] Applied Global DOMWatchdog Patch")
    
    # Patch StorageStateWatchdog to handle CDP client not initialized error
    from browser_use.browser.watchdogs.storage_state_watchdog import StorageStateWatchdog
    _original_on_save_storage_state = StorageStateWatchdog.on_SaveStorageStateEvent
    
    if not getattr(_original_on_save_storage_state, '_is_patched_storage', False):
        async def on_SaveStorageStateEvent(self, event):
            """
            Patched storage state save handler with CDP initialization check.
            Gracefully handles the case when CDP client is not initialized yet.
            """
            try:
                # 首先检查 browser_session 是否存在
                if not hasattr(self, 'browser_session') or self.browser_session is None:
                    logger.warning("[WARNING] Browser session not available, skipping storage state save")
                    return
                
                # 尝试获取或创建 CDP session 来验证连接（这会触发初始化检查）
                try:
                    session = await self.browser_session.get_or_create_cdp_session(target_id=None)
                    if session is None:
                        logger.warning("[WARNING] Failed to get CDP session, skipping storage state save")
                        return
                except AssertionError as e:
                    if 'Root CDP client not initialized' in str(e):
                        logger.warning("[WARNING] Root CDP client not initialized, skipping storage state save")
                        return
                    raise
                except AttributeError as e:
                    if '_cdp_client_root' in str(e):
                        logger.warning("[WARNING] CDP client root attribute not available, skipping storage state save")
                        return
                    raise
                
                # 如果 CDP 已初始化，正常执行保存操作
                return await _original_on_save_storage_state(self, event)
                
            except AssertionError as e:
                if 'Root CDP client not initialized' in str(e):
                    logger.warning("[WARNING] StorageStateWatchdog: CDP client not initialized, skipping save")
                    return
                raise
            except AttributeError as e:
                if '_cdp_client_root' in str(e) or 'browser_session' in str(e):
                    logger.warning(f"[WARNING] StorageStateWatchdog: Missing required attribute, skipping save: {e}")
                    return
                raise
            except Exception as e:
                # 记录错误但不抛出，避免影响主要流程
                logger.warning(f"[WARNING] StorageStateWatchdog failed to save storage state: {e}")
                return
        
        on_SaveStorageStateEvent._is_patched_storage = True
        StorageStateWatchdog.on_SaveStorageStateEvent = on_SaveStorageStateEvent
        logger.info("[OK] Applied Global StorageStateWatchdog Patch")
    
    # Patch BrowserStartEvent handler to increase timeout
    try:
        from browser_use.browser.session import BrowserSession
        _original_on_browser_start = None
        
        # 检查是否存在 on_BrowserStartEvent 方法
        if hasattr(BrowserSession, 'on_BrowserStartEvent'):
            _original_on_browser_start = BrowserSession.on_BrowserStartEvent
            
            if not getattr(_original_on_browser_start, '_is_patched_start', False):
                async def on_BrowserStartEvent(self, event):
                    """
                    Patched browser start event handler with increased timeout and better error handling.
                    """
                    try:
                        # 使用更长的超时时间等待浏览器启动（30秒）
                        result = await asyncio.wait_for(
                            _original_on_browser_start(self, event),
                            timeout=30.0  # 增加超时到30秒
                        )
                        return result
                    except asyncio.TimeoutError:
                        logger.error("[ERROR] Browser start timeout after 30 seconds")
                        raise
                    except Exception as e:
                        error_str = str(e)
                        # 检查是否是 JSON 解析错误
                        if 'JSONDecodeError' in error_str or 'Expecting value' in error_str:
                            logger.error(f"[ERROR] Browser start failed: JSON parsing error - {e}")
                            logger.error("[ERROR] This usually means the browser CDP endpoint is not ready or returned invalid data")
                            logger.error("[ERROR] Possible causes:")
                            logger.error("[ERROR] 1. Browser is still starting up")
                            logger.error("[ERROR] 2. CDP endpoint is not accessible")
                            logger.error("[ERROR] 3. Browser crashed during startup")
                        else:
                            logger.error(f"[ERROR] Browser start failed: {e}")
                        raise
                
                on_BrowserStartEvent._is_patched_start = True
                BrowserSession.on_BrowserStartEvent = on_BrowserStartEvent
                logger.info("[OK] Applied Global BrowserStartEvent Patch")
    except Exception as browser_start_error:
        logger.warning(f"[WARNING] Failed to patch BrowserStartEvent: {browser_start_error}")

except Exception as e:
    logger.error(f"[ERROR] Failed to apply Global Watchdog patches: {e}")

# Patch Agent verdict
try:
    from browser_use.agent.service import Agent
    from browser_use.agent.message_manager.service import AgentOutput
    
    _original_judge_and_log = Agent._judge_and_log
    
    def _agent_output_getattr(self, name):
        if name == 'verdict':
            if hasattr(self, 'next_goal') and self.next_goal:
                if any(w in str(self.next_goal).lower() for w in ['complete', 'done', 'finished', 'success']): return True
            if hasattr(self, 'evaluation_previous_goal') and self.evaluation_previous_goal:
                if any(w in str(self.evaluation_previous_goal).lower() for w in ['success', 'complete']): return True
            return False
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    if not hasattr(AgentOutput, '__getattr__'):
        AgentOutput.__getattr__ = _agent_output_getattr
        
    async def _patched_judge_and_log(self):
        try:
            return await _original_judge_and_log(self)
        except AttributeError as e:
            if 'verdict' in str(e):
                return None
            raise
    Agent._judge_and_log = _patched_judge_and_log
except Exception:
    pass


# ============================================================================
# PART 2: Helper Classes
# ============================================================================

from langchain_core.callbacks import BaseCallbackHandler
from typing import Any

class RawResponseLogger(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any: pass
    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        try:
            generation = response.generations[0][0]
            logger.info(f"DEBUG: Raw LLM Response: {generation.text}")
        except: pass


# ============================================================================
# PART 3: Base Browser Agent
# ============================================================================

from browser_use import Agent, Controller
from browser_use.browser.profile import BrowserProfile

class BaseBrowserAgent:
    def __init__(self, execution_mode='text', enable_gif=True, case_name=None):
        self.execution_mode = 'text'
        self.enable_gif = enable_gif  # GIF录制开关
        self.case_name = case_name or "Adhoc Task"  # 用例名称
        
        # Load Config from DB
        from apps.requirement_analysis.models import AIModelConfig
        
        # Select Config (always use text mode config)
        role_name = 'browser_use_text'
        config_obj = AIModelConfig.objects.filter(role=role_name, is_active=True).first()
        
        model_config = {}
        if config_obj:
            model_config = {
                'api_key': config_obj.api_key,
                'base_url': config_obj.base_url,
                'model_name': config_obj.model_name,
                'provider': config_obj.model_type
            }
        
        self.api_key = model_config.get('api_key') or os.getenv('AUTH_TOKEN')
        self.base_url = model_config.get('base_url') or os.getenv('BASE_URL')
        self.model_name = model_config.get('model_name') or os.getenv('MODEL_NAME')
        self.provider = model_config.get('provider', 'openai')
        
        if not self.api_key:
            raise ValueError(f"No API Key found for mode: {execution_mode}")

        self.llm = ChatOpenAI(
            model=self.model_name,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.0,
            callbacks=[RawResponseLogger()]
        )
        
        # browser-use requirement
        try:
            object.__setattr__(self.llm, 'provider', self.provider)
            object.__setattr__(self.llm, 'model', self.model_name)
        except:
            if not hasattr(self.llm, '__pydantic_extra__') or self.llm.__pydantic_extra__ is None:
                self.llm.__pydantic_extra__ = {}
            self.llm.__pydantic_extra__['provider'] = self.provider
            self.llm.__pydantic_extra__['model'] = self.model_name
            
    def _format_action(self, action):
        try:
            action_dict = {}
            if hasattr(action, 'model_dump'): action_dict = action.model_dump()
            elif hasattr(action, '_action_dict'): action_dict = action._action_dict
            elif hasattr(action, '_dict'): action_dict = action._dict
            elif isinstance(action, dict): action_dict = action
            else: return str(action)

            if not action_dict: return "待机"

            descriptions = []
            for name, params in action_dict.items():
                if not params and name not in ['scroll_down', 'scroll_up', 'done']: continue
                
                if name in ['go_to_url', 'navigate']:
                    url = params.get('url') if isinstance(params, dict) else params
                    descriptions.append(f"访问: {url}")
                elif name in ['click_element', 'click']:
                    index = params.get('index') if isinstance(params, dict) else params
                    descriptions.append(f"点击[{index}]")
                elif name in ['input_text', 'input']:
                    text = params.get('text') if isinstance(params, dict) else None
                    descriptions.append(f"输入: '{text}'")
                elif name == 'switch_tab':
                    index = params.get('index', params)
                    descriptions.append(f"切换标签 {index}")
                elif name == 'open_new_tab':
                    url = params.get('url', params)
                    descriptions.append(f"新标签打开: {url}")
                elif name == 'done':
                    descriptions.append("任务完成")
                else:
                    descriptions.append(f"{name}")
            return " | ".join(descriptions)
        except:
            return "执行操作"

    async def analyze_task(self, task_description: str):
        try:
            prompt = f"Break down this task into steps: {task_description}. Return JSON list of strings."
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip() if hasattr(response, 'content') else str(response)
            
            steps = []
            try:
                import json
                match = re.search(r'(\[.*\])', content, re.DOTALL)
                if match: steps = json.loads(match.group(1))
            except: pass
            
            if not steps:
                steps = [s.strip() for s in task_description.split('\n') if s.strip()]
            
            # 彻底清理生成的步骤描述中的重复编号
            cleaned_steps = []
            for s in steps:
                desc = s
                while True:
                    match = re.match(r'^\s*\d+[\.\s、:]+(.*)', desc)
                    if not match: break
                    desc = match.group(1).strip()
                if desc:
                    cleaned_steps.append(desc)
                
            return [{'id': i+1, 'description': s, 'status': 'pending'} for i, s in enumerate(cleaned_steps)]
        except:
            return [{'id': 1, 'description': task_description, 'status': 'pending'}]

    def _create_browser_profile(self):
        # Default implementation, can be overridden
        chrome_path = None
        import platform

        system = platform.system()
        if system == 'Windows':
            paths = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.expanduser(r"~\AppData\Local\Google\Chrome\Application\chrome.exe")
            ]
            for p in paths:
                if os.path.exists(p):
                    chrome_path = p
                    break
        elif system == 'Linux':
            # Linux 系统常见的 Chrome 路径
            paths = [
                '/usr/bin/google-chrome',
                '/usr/bin/google-chrome-stable',
                '/usr/bin/chromium-browser',
                '/usr/bin/chromium',
                '/opt/google/chrome/chrome',
                '/snap/bin/chromium',
            ]
            for p in paths:
                if os.path.exists(p):
                    chrome_path = p
                    break

        # 基础性能优化参数
        extra_args = [
            '--disable-blink-features=AutomationControlled',
            '--disable-infobars', '--disable-notifications',
            '--disable-background-networking',
            '--disable-background-timer-throttling',
            '--disable-renderer-backgrounding',
            '--disable-backgrounding-occluded-windows',
            '--disable-extensions',
            '--disable-web-security',  # 允许跨域请求
            # 忽略 HTTPS 证书错误
            '--ignore-certificate-errors',
            '--ignore-ssl-errors',
            '--ignore-certificate-errors-spki-list',
        ]

        # 根据操作系统添加特定参数
        if system == 'Linux':
            # Linux 服务器环境（特别是无头环境）必需的参数
            extra_args.extend([
                '--no-sandbox',                    # Linux 必需：禁用沙箱
                '--disable-setuid-sandbox',        # Linux 必需：禁用 setuid 沙箱
                '--disable-dev-shm-usage',         # Linux 必需：使用 /tmp 而不是 /dev/shm
                '--disable-gpu',                   # 禁用 GPU 加速（服务器通常无 GPU）
                '--headless=new',                  # Linux 服务器使用无头模式
                '--disable-software-rasterizer',   # 禁用软件光栅化器
                '--remote-debugging-port=0',       # 使用随机可用端口
            ])
        else:
            # macOS 和 Windows 使用显示模式
            extra_args.extend([
                '--no-sandbox',                    # 兼容性
                '--disable-gpu',
                '--remote-debugging-port=9222',
            ])

        return BrowserProfile(
            headless=(system == 'Linux'),  # Linux 使用无头模式，其他系统使用显示模式
            disable_security=True,
            executable_path=chrome_path,
            args=extra_args,
            wait_for_network_idle_page_load_time=0.2,
            minimum_wait_page_load_time=0.05,
            wait_between_actions=0.1,
            enable_default_extensions=False
        )

    async def run_task(self, task_description: str, planned_tasks=None, callback=None, should_stop=None):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
            
        controller = Controller()
        _task_was_done = False

        @controller.action('Done')
        async def done(success: bool = True, text: str = ""):
            nonlocal _task_was_done
            _task_was_done = True
            return f"Finished: {text}"

        @controller.action('mark_task_complete')
        async def mark_task_complete(task_id: int):
            logger.info(f"[OK] Explicitly marking task {task_id} as completed")
            if callback:
                try:
                    data = {'task_id': int(task_id), 'status': 'completed'}
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    logger.warning(f"Failed to execute mark_task_complete callback: {e}")
            return f"Task {task_id} marked completed"

        # 构建强化版 Prompt
        final_task = task_description
        if planned_tasks:
            final_task += "\n\nIMPORTANT INSTRUCTION:\n"
            final_task += "You have a list of sub-tasks. Execute strictly in order.\n"
            final_task += "CRITICAL: MUST call 'mark_task_complete(task_id=...)' IMMEDIATELY after verifying each sub-task completion. NEVER skip this step. For every action you take, there MUST be a corresponding mark_task_complete call.\n"
            final_task += "IMPORTANT: If a sub-task (like opening a URL) is already fulfilled by the initial state, YOU MUST mark it complete in your VERY FIRST STEP.\n"
            final_task += "Sub-tasks (Execute in order):\n"
            cleaned_tasks = []
            for t in planned_tasks:
                desc = t['description']
                # 递归去除所有层级的重复序号，例如 "1. 1. xxx" -> "xxx"
                while True:
                    match = re.match(r'^\s*\d+[\.\s、:]+(.*)', desc)
                    if not match: break
                    desc = match.group(1).strip()
                cleaned_tasks.append(f"{t['id']}. {desc}")
            final_task += "\n".join(cleaned_tasks)

        # 极限效率版标记指令
        from datetime import datetime
        final_task += f"\n\nCURRENT TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        final_task += "\nCRITICAL PERFORMANCE & SYNC RULES:\n"
        final_task += "1. ACTION-TASK MAPPING: For EVERY sub-task that requires an action (click, input, select), you MUST call 'mark_task_complete(task_id=...)' in the SAME STEP as that action. DO NOT skip any task ID. Example: After clicking a button for task 8, immediately call mark_task_complete(task_id=8). IF YOU PERFORM MULTIPLE ACTIONS IN ONE STEP, YOU MUST CALL mark_task_complete FOR EACH CORRESPONDING TASK.\n"
        final_task += "2. NO JAVASCRIPT IN INPUT: When a task asks for a timestamp, YOU MUST compute the final string yourself (e.g., 'V8.01734892400').\n"
        final_task += "   - DO NOT output 'Date.now()' or '{{...}}' strings. Use the CURRENT TIME provided above to estimate a timestamp.\n"
        final_task += "3. DROPDOWN & MODAL ISOLATION: If an action (clicking a button/dropdown) triggers a UI change (modal opens/dropdown expands), YOU MUST STOP and WAIT for the next step to see the new elements. DO NOT attempt to interact with newly appeared elements (like dropdown options) in the same step as the click that opened them.\n"
        final_task += "4. ULTRALIGHT THINKING: Keep 'thinking' under 10 words. Just list next actions. Merge multiple INPUTS if they are on the same form, but NEVER merge a UI-opening click with its subsequent interaction. SPEED IS CRITICAL - respond as quickly as possible.\n"
        final_task += "5. RETRY LOGIC: If a previous 'save' or 'submit' failed (e.g., error toast), RE-VERIFY all fields. Re-select dropdowns and re-input text to ensure the form is complete. Often errors are caused by missing project selection.\n"
        final_task += "6. DO NOT REPEAT: If a task is complete, mark it and MOVE ON. Don't re-confirm unless the system requires it.\n"
        final_task += "7. VERIFICATION: Task 15/16 usually require checking the list. Ensure you are on the correct page and the new data is visible before marking complete.\n"
        
        if 'qwen' in self.model_name.lower() or 'deepseek' in self.model_name.lower():
            final_task += "8. EXTREMELY MINIMIZE output tokens for speed. Keep responses as short as possible while maintaining accuracy.\n"

        # 核心修复: 清理 task 长文本中的 URL，防止中文标点紧贴 URL 导致 browser-use 解析错误
        # 例如 "http://localhost:3000，" -> "http://localhost:3000 "
        try:
             # 在中文标点前加空格，避免它们成为 URL 的一部分
             final_task = re.sub(r'(https?://[^\s\u4e00-\u9fa5]+?)(?=[，；。、！])', r'\1 ', final_task)
             logger.info(f"[OPT] Optimized task description for URL extraction")
        except:
             pass

        browser_profile = self._create_browser_profile()
        
        agent = Agent(
            task=final_task,
            llm=self.llm,
            controller=controller,
            browser_profile=browser_profile,
            use_vision=False,
            max_actions_per_step=10, # 增加步进密度，减少总步骤数，降低超时风险
            max_retries=1, # 减少重试次数以提高速度 (从2改为1)
            max_failures=2, # 减少最大失败次数，避免过长等待 (从默认3改为2)
            llm_timeout=45, # 设置LLM调用超时为45秒
            step_timeout=90, # 设置每步超时为90秒
            generate_gif=self.enable_gif, # 根据开关决定是否生成GIF
        )
        agent._task_was_done = False

        # Callback helper - 添加任务标记跟踪
        last_processed_step = 0
        last_marked_task_id = 0  # 跟踪上一次标记的任务ID
        completed_task_ids = set()  # 跟踪已完成的任务ID
        
        async def on_step_end(agent_instance):
            nonlocal last_processed_step, last_marked_task_id, completed_task_ids

            if should_stop:
                do_stop = await should_stop() if asyncio.iscoroutinefunction(should_stop) else should_stop()
                if do_stop: raise KeyboardInterrupt("User requested stop")

            if _task_was_done:
                raise KeyboardInterrupt("Done")

            history = getattr(agent_instance, 'history', [])
            if hasattr(history, 'history'): history = history.history

            if len(history) > last_processed_step:
                for i in range(last_processed_step, len(history)):
                    step = history[i]
                    # Log logic here
                    try:
                        actions = []
                        if hasattr(step, 'model_output') and hasattr(step.model_output, 'action'):
                            raw = step.model_output.action
                            actions = raw if isinstance(raw, list) else [raw]

                        # 检查这一步是否调用了mark_task_complete
                        step_has_task_complete = False
                        step_marked_task_id = None
                        for action in actions:
                            action_dict = action.model_dump() if hasattr(action, 'model_dump') else getattr(action, '_action_dict', {})
                            if 'mark_task_complete' in action_dict:
                                step_has_task_complete = True
                                step_marked_task_id = action_dict['mark_task_complete'].get('task_id')
                                if step_marked_task_id:
                                    completed_task_ids.add(int(step_marked_task_id))
                                    last_marked_task_id = step_marked_task_id
                                break

                        # 检查这一步是否有实际操作（非mark_task_complete的操作）
                        has_real_action = False
                        has_task_completing_action = False  # 检查是否有可以表示任务完成的操作
                        for action in actions:
                            action_dict = action.model_dump() if hasattr(action, 'model_dump') else getattr(action, '_action_dict', {})
                            for key in action_dict.keys():
                                if key not in ['mark_task_complete', 'done']:
                                    has_real_action = True
                                    # 只有某些操作可以表示任务完成（排除中间操作如scroll, wait等）
                                    # 这些操作通常表示任务的核心动作已完成
                                    if key in ['click', 'fill', 'submit', 'select', 'navigate', 'go']:
                                        has_task_completing_action = True
                                    break
                            if has_real_action:
                                break

                        action_str = " | ".join([self._format_action(a) for a in actions])
                        log_content = f"\n[Step {i+1}]\n执行: {action_str}\n"

                        if callback:
                            if asyncio.iscoroutinefunction(callback): await callback({'type': 'log', 'content': log_content})
                            else: callback({'type': 'log', 'content': log_content})

                        # 关键修复：如果这一步有可以表示任务完成的操作但没有调用mark_task_complete，
                        # 且planned_tasks中下一个未标记的任务ID应该被标记
                        # 注意：scroll、wait等中间操作不应该触发自动标记
                        if has_task_completing_action and not step_has_task_complete and planned_tasks:
                            # 找出下一个应该标记的任务ID
                            next_expected_task_id = last_marked_task_id + 1
                            if next_expected_task_id <= len(planned_tasks):
                                # 检查这个任务是否还没有被标记
                                task_already_marked = False
                                for task in planned_tasks:
                                    if task['id'] == next_expected_task_id and task.get('status') == 'completed':
                                        task_already_marked = True
                                        break

                                if not task_already_marked:
                                    # 自动补充标记这个任务
                                    logger.warning(f"[WARNING] Auto-fixing: Step {i+1} had actions but no mark_task_complete. Auto-marking task {next_expected_task_id} as completed.")
                                    data = {'task_id': int(next_expected_task_id), 'status': 'completed'}
                                    if asyncio.iscoroutinefunction(callback):
                                        await callback(data)
                                    else:
                                        callback(data)
                                    last_marked_task_id = next_expected_task_id
                                    completed_task_ids.add(int(next_expected_task_id))
                                    
                                    # 检查是否所有任务都已完成
                                    if planned_tasks and len(completed_task_ids) >= len(planned_tasks):
                                        logger.info(f"[OK] All {len(planned_tasks)} tasks completed, auto-terminating")
                                        raise KeyboardInterrupt("All tasks completed")

                    except KeyboardInterrupt:
                        raise  # 重新抛出 KeyboardInterrupt
                    except Exception as e:
                        logger.warning(f"[WARNING] Error in on_step_end processing: {e}")
                
                last_processed_step = len(history)
                
                # 最终检查：如果所有任务都已完成，自动结束
                if planned_tasks and len(completed_task_ids) >= len(planned_tasks):
                    logger.info(f"[OK] All {len(planned_tasks)} tasks completed after step {last_processed_step}, auto-terminating")
                    raise KeyboardInterrupt("All tasks completed")

        try:
            # Try to pass callback
            import inspect
            sig = inspect.signature(agent.run)
            if 'on_step_end' in sig.parameters:
                await agent.run(max_steps=100, on_step_end=on_step_end)
            else:
                await agent.run(max_steps=100)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            raise

        # 在任务结束时检查不一致的任务状态
        history = getattr(agent, 'history', [])
        if history:
            logger.info("[CHECK] Performing final task status consistency check")
            # 检查是否有任务执行了但未标记完成
            executed_tasks_info = self._find_executed_tasks(history)
            if executed_tasks_info and executed_tasks_info.get('unmarked_actions'):
                logger.warning(f"[WARNING] Found {executed_tasks_info['executed_actions']} executed actions, but only {len(executed_tasks_info['marked_tasks'])} tasks were explicitly marked complete")
                logger.warning(f"[WARNING] Unmarked actions: {executed_tasks_info['unmarked_actions']}")
                logger.warning("[WARNING] This indicates the AI agent did not follow the 'mark_task_complete' rule properly.")

        return history

    def _find_executed_tasks(self, history):
        """
        通过分析执行历史找出已执行但未标记完成的任务
        """
        if not history or not hasattr(history, 'steps'):
            return []

        executed_actions = {}  # 已执行的操作类型和索引，以及对应的步骤
        marked_tasks = set()   # 已标记完成的任务ID

        # 分析执行历史
        for step_idx, step in enumerate(getattr(history, 'steps', [])):
            # 检查每一步中的actions
            actions = getattr(step, 'actions', [])
            for action in actions:
                # 记录已执行的操作
                if hasattr(action, 'input'):
                    action_key = f"input_{action.input.index}"
                    executed_actions[action_key] = {
                        'step': step_idx,
                        'action': 'input',
                        'index': action.input.index
                    }
                elif hasattr(action, 'click'):
                    action_key = f"click_{action.click.index}"
                    executed_actions[action_key] = {
                        'step': step_idx,
                        'action': 'click',
                        'index': action.click.index
                    }
                elif hasattr(action, 'switch_tab'):
                    action_key = f"switch_tab_{action.switch_tab.tab_id}"
                    executed_actions[action_key] = {
                        'step': step_idx,
                        'action': 'switch_tab',
                        'tab_id': action.switch_tab.tab_id
                    }

                # 记录已标记完成的任务
                if hasattr(action, 'mark_task_complete'):
                    marked_tasks.add(action.mark_task_complete.task_id)

        # 理想情况下应该有一个映射机制来关联操作和任务，但由于我们没有这个映射，
        # 我们只能记录未标记完成的执行操作作为调试信息
        unmarked_actions = []
        for action_key, action_info in executed_actions.items():
            unmarked_actions.append({
                'action': action_info['action'],
                'step': action_info['step'],
                'details': action_key
            })

        return {
            'marked_tasks': list(marked_tasks),
            'executed_actions': len(executed_actions),
            'unmarked_actions': unmarked_actions
        }

    async def run_full_process(self, task_description: str, analysis_callback=None, step_callback=None, should_stop=None):
        planned_tasks = await self.analyze_task(task_description)
        if analysis_callback:
            if asyncio.iscoroutinefunction(analysis_callback): await analysis_callback(planned_tasks)
            else: analysis_callback(planned_tasks)
            
        return await self.run_task(task_description, planned_tasks, step_callback, should_stop)
