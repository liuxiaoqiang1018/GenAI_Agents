# 第6课 - Java 端实现参考（伪代码）

## 整体架构

```
Controller → MultiAgentOrchestrator → Coordinator → Agent[] → ResultMerger
                                          ↓
                                    ProfileAnalyzer
```

对应 Spring Boot 微服务风格，每个 Agent 是一个 Service。

---

## 1. 状态定义（对应 Python 的 AcademicState）

```java
/**
 * 对应 Python:
 *   class AcademicState(TypedDict):
 *       messages: Annotated[List, add]
 *       profile: Annotated[Dict, dict_reducer]
 *       ...
 */
@Data
public class AcademicState {
    private List<Message> messages = new ArrayList<>();
    private StudentProfile profile;
    private Calendar calendar;
    private TaskList tasks;
    private Map<String, Object> results = new ConcurrentHashMap<>(); // 线程安全，支持并行写入

    /** 深度合并结果（对应 Python 的 dict_reducer） */
    public synchronized void mergeResults(Map<String, Object> newResults) {
        for (var entry : newResults.entrySet()) {
            Object existing = results.get(entry.getKey());
            if (existing instanceof Map && entry.getValue() instanceof Map) {
                ((Map) existing).putAll((Map) entry.getValue());
            } else {
                results.put(entry.getKey(), entry.getValue());
            }
        }
    }
}

@Data
public class StudentProfile {
    private String name;        // "张明"
    private String grade;       // "大三"
    private String major;       // "计算机科学"
    private String learningStyle; // "视觉型"
    private List<String> strongSubjects;
    private List<String> weakSubjects;
    private LearningPreference preference;
}

@Data
public class LearningPreference {
    private String bestTimeSlot;    // "上午9-12点"
    private String focusDuration;   // "45分钟"
    private String preferredMethod; // "思维导图 + 做题"
}
```

---

## 2. LLM 调用层（对应 Python 的 ChatOpenAI / OpenAI）

```java
/**
 * 对应 Python:
 *   llm = ChatOpenAI(model=..., base_url=..., api_key=...)
 *   response = llm.invoke([HumanMessage(content=prompt)])
 *
 * 用 OkHttp/RestTemplate 调用 OpenAI 兼容接口
 */
@Service
public class LLMClient {

    @Value("${llm.base-url}")
    private String baseUrl;

    @Value("${llm.api-key}")
    private String apiKey;

    @Value("${llm.model-name}")
    private String modelName;

    private final RestTemplate restTemplate;

    public String chat(String prompt) {
        return chat(prompt, 0.5);
    }

    public String chat(String prompt, double temperature) {
        HttpHeaders headers = new HttpHeaders();
        headers.setBearerAuth(apiKey);
        headers.setContentType(MediaType.APPLICATION_JSON);
        // 注意：某些第三方 API 需要自定义 User-Agent
        headers.set("User-Agent", "Mozilla/5.0");

        Map<String, Object> body = Map.of(
            "model", modelName,
            "temperature", temperature,
            "messages", List.of(
                Map.of("role", "user", "content", prompt)
            )
        );

        ResponseEntity<Map> resp = restTemplate.exchange(
            baseUrl + "/chat/completions",
            HttpMethod.POST,
            new HttpEntity<>(body, headers),
            Map.class
        );

        // 提取 choices[0].message.content
        List<Map> choices = (List<Map>) resp.getBody().get("choices");
        Map message = (Map) choices.get(0).get("message");
        return (String) message.get("content");
    }
}
```

---

## 3. Agent 基类（对应 Python 的节点函数）

```java
/**
 * 对应 Python:
 *   def planner_node(state: AcademicState) -> Dict:
 *       prompt = f"..."
 *       response = llm.invoke([HumanMessage(content=prompt)])
 *       return {"results": {"planner_output": response}}
 *
 * Java 中每个 Agent 是一个 Function<AcademicState, Map<String, Object>>
 */
public interface Agent {
    String getName();
    Map<String, Object> execute(AcademicState state);
}

@Service
public abstract class BaseAgent implements Agent {
    @Autowired
    protected LLMClient llm;

    /** 子类实现：构建 Prompt */
    protected abstract String buildPrompt(AcademicState state);

    /** 子类实现：结果 key，如 "planner_output" */
    protected abstract String getResultKey();

    @Override
    public Map<String, Object> execute(AcademicState state) {
        String prompt = buildPrompt(state);
        String response = llm.chat(prompt);
        return Map.of(getResultKey(), response);
    }
}
```

---

## 4. 三个 Agent 实现

```java
/**
 * 对应 Python: def planner_node(state) -> Dict
 */
@Service("PLANNER")
public class PlannerAgent extends BaseAgent {

    @Override
    public String getName() { return "PLANNER"; }

    @Override
    protected String getResultKey() { return "planner_output"; }

    @Override
    protected String buildPrompt(AcademicState state) {
        String query = state.getMessages().getLast().getContent();
        String profileAnalysis = (String) state.getResults()
            .getOrDefault("profile_analysis", "");

        return String.format("""
            你是学习计划Agent。根据学生的日程和任务，生成具体的时间安排。

            学生特征：%s
            当前日程：%s
            待办任务：%s
            用户请求：%s

            请生成按天排列的学习计划。考虑：
            1. 避开已有日程
            2. 利用学生的最佳学习时段
            3. 高优先级任务优先安排
            4. 每次学习不超过学生的专注时长
            """,
            profileAnalysis,
            toJson(state.getCalendar()),
            toJson(state.getTasks()),
            query
        );
    }
}

/**
 * 对应 Python: def notewriter_node(state) -> Dict
 */
@Service("NOTEWRITER")
public class NoteWriterAgent extends BaseAgent {

    @Override
    public String getName() { return "NOTEWRITER"; }

    @Override
    protected String getResultKey() { return "notewriter_output"; }

    @Override
    protected String buildPrompt(AcademicState state) {
        String query = state.getMessages().getLast().getContent();
        String profileAnalysis = (String) state.getResults()
            .getOrDefault("profile_analysis", "");

        return String.format("""
            你是学习笔记Agent。根据学生的学习风格生成结构化学习材料。
            学生特征：%s
            用户请求：%s
            """, profileAnalysis, query);
    }
}

/**
 * 对应 Python: def advisor_node(state) -> Dict
 */
@Service("ADVISOR")
public class AdvisorAgent extends BaseAgent {

    @Override
    public String getName() { return "ADVISOR"; }

    @Override
    protected String getResultKey() { return "advisor_output"; }

    @Override
    protected String buildPrompt(AcademicState state) {
        String query = state.getMessages().getLast().getContent();
        String profileAnalysis = (String) state.getResults()
            .getOrDefault("profile_analysis", "");

        return String.format("""
            你是学习顾问Agent。提供个性化学习建议。
            学生特征：%s
            学生信息：%s
            用户请求：%s
            """, profileAnalysis, toJson(state.getProfile()), query);
    }
}
```

---

## 5. 协调器（对应 Python 的 coordinator_node）

```java
/**
 * 对应 Python:
 *   def coordinator_node(state) -> Dict:
 *       prompt = f"...分析请求，决定需要哪些Agent..."
 *       response = llm.invoke(...)
 *       analysis = json.loads(response)
 *       return {"results": {"coordinator_analysis": analysis}}
 */
@Service
public class CoordinatorService {

    @Autowired
    private LLMClient llm;

    @Autowired
    private ObjectMapper objectMapper;

    public CoordinatorDecision analyze(AcademicState state) {
        String query = state.getMessages().getLast().getContent();

        String prompt = String.format("""
            你是学术辅助系统的协调器。判断请求是否与学术相关，决定需要哪些Agent。

            规则：
            - 非学术问题：返回 {"required_agents": [], "direct_answer": "直接回答"}
            - 学术问题：返回 {"required_agents": ["PLANNER"], "reasoning": "原因"}

            可用Agent：PLANNER（日程计划）、NOTEWRITER（学习笔记）、ADVISOR（学习建议）

            学生信息：%s
            待办任务：%s
            用户请求：%s
            """,
            toJson(state.getProfile()),
            toJson(state.getTasks()),
            query
        );

        String response = llm.chat(prompt, 0);
        return parseDecision(response);
    }

    /**
     * 容错解析（对应 Python 中的大括号深度匹配逻辑）
     */
    private CoordinatorDecision parseDecision(String response) {
        // 1. 直接尝试解析
        try {
            return objectMapper.readValue(response, CoordinatorDecision.class);
        } catch (Exception ignored) {}

        // 2. 提取第一个完整 JSON
        int start = response.indexOf('{');
        if (start >= 0) {
            int depth = 0;
            for (int i = start; i < response.length(); i++) {
                if (response.charAt(i) == '{') depth++;
                if (response.charAt(i) == '}') depth--;
                if (depth == 0) {
                    try {
                        return objectMapper.readValue(
                            response.substring(start, i + 1),
                            CoordinatorDecision.class
                        );
                    } catch (Exception ignored) {}
                    break;
                }
            }
        }

        // 3. 兜底：从文本识别 Agent 名称
        List<String> agents = new ArrayList<>();
        for (String name : List.of("PLANNER", "NOTEWRITER", "ADVISOR")) {
            if (response.toUpperCase().contains(name)) {
                agents.add(name);
            }
        }
        return new CoordinatorDecision(
            agents.isEmpty() ? List.of("PLANNER") : agents,
            "从文本提取", null
        );
    }
}

@Data
@AllArgsConstructor
@NoArgsConstructor
public class CoordinatorDecision {
    private List<String> requiredAgents = new ArrayList<>();
    private String reasoning;
    private String directAnswer; // 非学术问题时的直接回答
}
```

---

## 6. 档案分析（对应 Python 的 profile_analyzer_node）

```java
@Service
public class ProfileAnalyzer {

    @Autowired
    private LLMClient llm;

    public String analyze(StudentProfile profile) {
        String prompt = String.format("""
            分析学生档案，用一段简洁的中文总结学习特征。
            档案：%s
            """, toJson(profile));

        return llm.chat(prompt);
    }
}
```

---

## 7. 编排器（对应 Python 的 build_workflow + invoke）

```java
/**
 * 这是整个系统的核心，对应 Python 的：
 *   workflow = StateGraph(AcademicState)
 *   workflow.add_node("coordinator", coordinator_node)
 *   workflow.add_conditional_edges(...)
 *   app = workflow.compile()
 *   result = app.invoke(initial_state)
 *
 * Java 不用图框架，直接写编排逻辑（更清晰）
 */
@Service
public class MultiAgentOrchestrator {

    @Autowired
    private CoordinatorService coordinator;

    @Autowired
    private ProfileAnalyzer profileAnalyzer;

    @Autowired
    private Map<String, Agent> agentMap; // Spring 自动注入所有 Agent 实现

    @Autowired
    private TaskExecutor taskExecutor; // Spring 线程池

    public String process(AcademicState state) {

        // ========== 第1步：协调器 ==========
        CoordinatorDecision decision = coordinator.analyze(state);

        // 非学术问题 → 直接返回（短路）
        if (decision.getRequiredAgents().isEmpty()) {
            return decision.getDirectAnswer() != null
                ? decision.getDirectAnswer()
                : "我是学习助手，这个问题超出了我的范围。";
        }

        // ========== 第2步：档案分析 ==========
        String profileAnalysis = profileAnalyzer.analyze(state.getProfile());
        state.mergeResults(Map.of("profile_analysis", profileAnalysis));

        // ========== 第3步：条件路由 + 并行执行 Agent ==========
        /**
         * 对应 Python 的 add_conditional_edges:
         *   workflow.add_conditional_edges(
         *       "profile_analyzer",
         *       route_to_agents,
         *       ["planner", "notewriter", "advisor"],
         *   )
         *
         * Java 用 CompletableFuture 实现并行
         */
        List<CompletableFuture<Map<String, Object>>> futures = decision
            .getRequiredAgents()
            .stream()
            .map(agentName -> {
                Agent agent = agentMap.get(agentName);
                if (agent == null) return null;
                // 每个 Agent 异步执行
                return CompletableFuture.supplyAsync(
                    () -> agent.execute(state), taskExecutor
                );
            })
            .filter(Objects::nonNull)
            .toList();

        // 等待所有 Agent 完成
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();

        // ========== 第4步：合并结果 ==========
        /**
         * 对应 Python 的 Annotated[Dict, dict_reducer]
         * 多个 Agent 的结果深度合并到 state.results
         */
        for (var future : futures) {
            state.mergeResults(future.join());
        }

        // ========== 第5步：组装输出 ==========
        return buildFinalOutput(state.getResults());
    }

    private String buildFinalOutput(Map<String, Object> results) {
        StringBuilder sb = new StringBuilder();

        if (results.containsKey("planner_output")) {
            sb.append("【学习计划】\n").append(results.get("planner_output")).append("\n\n---\n\n");
        }
        if (results.containsKey("notewriter_output")) {
            sb.append("【学习材料】\n").append(results.get("notewriter_output")).append("\n\n---\n\n");
        }
        if (results.containsKey("advisor_output")) {
            sb.append("【学习建议】\n").append(results.get("advisor_output"));
        }

        return sb.length() > 0 ? sb.toString() : "暂无结果";
    }
}
```

---

## 8. Controller 层

```java
@RestController
@RequestMapping("/api/atlas")
public class AtlasController {

    @Autowired
    private MultiAgentOrchestrator orchestrator;

    @Autowired
    private StudentDataService studentDataService; // 从数据库读取学生数据

    @PostMapping("/ask")
    public ResponseEntity<AskResponse> ask(
            @RequestHeader("X-Student-Id") String studentId,
            @RequestBody AskRequest request) {

        // 从数据库加载学生数据（demo 中是硬编码）
        StudentProfile profile = studentDataService.getProfile(studentId);
        Calendar calendar = studentDataService.getCalendar(studentId);
        TaskList tasks = studentDataService.getTasks(studentId);

        AcademicState state = new AcademicState();
        state.setProfile(profile);
        state.setCalendar(calendar);
        state.setTasks(tasks);
        state.getMessages().add(new Message("user", request.getQuestion()));

        String result = orchestrator.process(state);

        return ResponseEntity.ok(new AskResponse(result));
    }
}
```

---

## Python → Java 概念对照表

| Python（LangGraph） | Java（Spring Boot） | 说明 |
|---------------------|---------------------|------|
| `StateGraph(AcademicState)` | `MultiAgentOrchestrator` | 编排器，管理流程 |
| `workflow.add_node("planner", fn)` | `@Service("PLANNER") class PlannerAgent` | 注册节点/Agent |
| `workflow.add_edge(A, B)` | 方法内顺序调用 | 固定边 = 顺序执行 |
| `workflow.add_conditional_edges()` | `if/switch` + `CompletableFuture` | 条件路由 + 并行 |
| `Annotated[Dict, dict_reducer]` | `ConcurrentHashMap` + `synchronized mergeResults()` | 并发安全的状态合并 |
| `app.invoke(state)` | `orchestrator.process(state)` | 执行入口 |
| `llm.invoke([HumanMessage(...)])` | `llmClient.chat(prompt)` | LLM 调用 |
| `json.loads()` + 容错 | `ObjectMapper` + 大括号深度匹配 | JSON 解析 |
| `.env` + `load_dotenv()` | `application.yml` + `@Value` | 配置管理 |
| `python main.py`（交互模式） | REST API `/api/atlas/ask` | 接入方式 |

---

## 关键差异总结

1. **不需要图框架**：Java 用 `if-else` + `CompletableFuture` 就能实现 LangGraph 的条件路由和并行执行，代码更直白
2. **并行用线程池**：Python 的 LangGraph 内部用 asyncio，Java 用 `CompletableFuture` + Spring `TaskExecutor`
3. **状态合并用锁**：Python 的 `dict_reducer` 在 Java 中对应 `ConcurrentHashMap` + `synchronized`
4. **Agent 注册用 Spring IoC**：`@Service("PLANNER")` 注册后，`Map<String, Agent>` 自动注入，等价于 `workflow.add_node()`
5. **配置用 application.yml**：不需要 `.env`，Spring Boot 原生支持多环境配置
