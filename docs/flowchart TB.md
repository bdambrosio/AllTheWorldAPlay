<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>

<div class="mermaid">
flowchart TB
    subgraph CognitiveProcess["Cognitive Cycle (async cognitive_cycle)"]
        direction TB
        CC[cognitive_cycle] --> UNM[Update Narrative & Memory]
        UNM -->|"if no focus_goal\nor goal satisfied"| GSG[Generate & Select Goal]
        GSG -->|"if no active task\nor task satisfied"| GTP[Generate & Select Task Plan]
        GTP --> STP[Step Task]
        STP -->|"for active tasks"| GAA[Generate & Select Act Alternatives]
        GAA --> AOA[Act on Action]
    end

    subgraph DriveSystem["Drive System"]
        direction TB
        DS[DriveSignalManager] --> DCP[Detect & Cluster Patterns]
        DCP --> SigC["SignalClusters (scored)"]
        SigC --> ES[EmotionalStance]
    end

    subgraph MemorySystem["Memory System"]
        direction TB
        SM[StructuredMemory] --> CM[Concrete Memories]
        SM --> AM[Abstract Memories]
        MR[MemoryRetrieval] --> QCM[Query Concrete Memories]
        MR --> QAM[Query Abstract Memories]
        MC[MemoryConsolidator] --> UNS[Update Narrative Summary]
    end

    subgraph PerceptionSystem["Perception System"]
        direction TB
        PS[PerceptualState] --> PI[Perceptual Inputs]
        PI --> FP[Filter & Process]
    end

    subgraph SocialSystem["Social System"]
        direction TB
        KAM[KnownActorManager] --> KAC[Known Actor Models]
        KAC --> DLG[Dialog Management]
    end

    %% External method connections
    LOOK[look\n<LLM call>] --> PI
    LOOK -.-> |updates| look_percept
    
    GGAS[generate_goal_alternatives\n<LLM call>] --> SigC
    GGAS -.-> |creates| goals
    GGAS -.-> |uses| look_percept
    
    GTPA[generate_task_plan\n<LLM call>] --> |uses| focus_goal
    GTPA -.-> |creates| tasks
    
    GACTA[generate_acts\n<LLM call>] --> |uses| focus_task
    GACTA -.-> |creates| actions
    
    %% System interactions
    PI --> DS
    PS --> UNM
    DS --> GSG
    SM --> GSG
    SM --> GTP
    KAM --> AOA
    DLG --> AOA
    
    %% Main data structures
    classDef datastructure fill:#f9f,stroke:#333,stroke-width:1px
    
    class goals,tasks,actions,focus_goal,focus_task,look_percept datastructure
    
    %% LLM calls
    classDef llmCall fill:#bbf,stroke:#333,stroke-width:2px
    
    class LOOK,GGAS,GTPA,GACTA llmCall
    
    %% Systems
    classDef system fill:#dfd,stroke:#333,stroke-width:1px
    
    class DriveSystem,MemorySystem,PerceptionSystem,SocialSystem system
    
    %% Important processes
    classDef process fill:#ffd,stroke:#333,stroke-width:1px
    
    class CognitiveProcess process
</div>

