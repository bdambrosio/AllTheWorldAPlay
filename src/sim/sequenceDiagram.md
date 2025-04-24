sequenceDiagram
    participant User
    participant CognitiveCycle as Character.cognitive_cycle
    participant MemorySystem as Memory & Consolidation
    participant GoalTaskSystem as Goal/Task Generation & Selection
    participant ActionSystem as Action Generation & Execution
    participant WorldContext as Context Interaction
    participant DialogSystem as Dialog Management

    User->>CognitiveCycle: Start Cycle

    CognitiveCycle->>MemorySystem: update_cognitive_model()
    MemorySystem-->>MemorySystem: LLM: Update Narrative & Relationships
    MemorySystem-->>CognitiveCycle: Updated Narrative/Models

    alt Focus Goal Exists?
        CognitiveCycle->>GoalTaskSystem: clear_goal_if_satisfied(focus_goal)
        GoalTaskSystem-->>GoalTaskSystem: LLM: test_termination(goal)
        GoalTaskSystem-->>CognitiveCycle: Goal Satisfied? (T/F)
        Note over CognitiveCycle: If Satisfied, focus_goal = None
    end

    alt No Focus Goal?
        CognitiveCycle->>GoalTaskSystem: generate_goal_alternatives()
        GoalTaskSystem-->>GoalTaskSystem: LLM: Generate Goals
        GoalTaskSystem-->>CognitiveCycle: Goal Alternatives

        CognitiveCycle->>GoalTaskSystem: request_goal_choice(alternatives)
        opt Needs Precondition Check
             GoalTaskSystem->>GoalTaskSystem: admissible_goals(alternatives)
             GoalTaskSystem-->>GoalTaskSystem: LLM: Check Preconditions
             GoalTaskSystem-->>GoalTaskSystem: Filtered Goals
        end
        GoalTaskSystem-->>CognitiveCycle: Selected Focus Goal (focus_goal)
        Note over CognitiveCycle: Other goals removed.
    end

    alt Focus Goal Has No Plan?
        CognitiveCycle->>GoalTaskSystem: generate_task_plan(focus_goal)
        GoalTaskSystem-->>GoalTaskSystem: LLM(default_ask): Generate Task Sequence
        GoalTaskSystem-->>CognitiveCycle: Task Plan for Goal
    end

    CognitiveCycle->>GoalTaskSystem: request_task_choice(plan)
    GoalTaskSystem-->>CognitiveCycle: Selected Focus Task (Pushed to Stack)

    CognitiveCycle->>CognitiveCycle: step_task()
    loop Until Task Done or Max Acts
        CognitiveCycle->>ActionSystem: generate_acts(focus_task)
        ActionSystem-->>ActionSystem: LLM(default_ask): Generate Action Alternatives
        ActionSystem-->>CognitiveCycle: Action Alternatives

        CognitiveCycle->>ActionSystem: request_act_choice(alternatives)
        ActionSystem-->>CognitiveCycle: Selected Focus Action

        CognitiveCycle->>ActionSystem: act_on_action(focus_action, focus_task)
        ActionSystem->>ActionSystem: acts(action, ...)
        alt Action Mode == 'Do'
            ActionSystem->>WorldContext: context.do(self, action_arg)
            WorldContext-->>WorldContext: LLM: Determine Consequences
            WorldContext-->>WorldContext: LLM: World Updates
            WorldContext-->>WorldContext: LLM: Character Updates
            WorldContext-->>ActionSystem: Consequences, World Updates, Char Updates
            ActionSystem-->>ActionSystem: Update State & Add Percepts
        else Action Mode == 'Say'
            ActionSystem->>DialogSystem: hear(target, message) / tell(target, message)
            DialogSystem-->>DialogSystem: LLM: natural_dialog_end()?
            DialogSystem-->>DialogSystem: LLM: generate_dialog_turn()
            DialogSystem-->>DialogSystem: LLM: update_commitments()? (if dialog ends)
            DialogSystem-->>ActionSystem: (May trigger recursive act_on_action via hear/think)
        else Action Mode == 'Think'
            ActionSystem->>DialogSystem: think(message)
            DialogSystem-->>DialogSystem: LLM: natural_dialog_end()?
            DialogSystem-->>DialogSystem: LLM: generate_dialog_turn()
            DialogSystem-->>DialogSystem: LLM: update_commitments()? (if dialog ends)
            DialogSystem-->>ActionSystem: (May trigger recursive act_on_action via think)
        else Action Mode == 'Look'/'Move'
             ActionSystem->>ActionSystem: Perform Look/Move
             ActionSystem->>ActionSystem: Update Perceptual State (No LLM)
        end
        ActionSystem-->>CognitiveCycle: Action Completed

        CognitiveCycle->>GoalTaskSystem: clear_task_if_satisfied(focus_task)
        GoalTaskSystem-->>GoalTaskSystem: LLM: test_termination(task)
        GoalTaskSystem-->>CognitiveCycle: Task Satisfied? (T/F)
        Note over CognitiveCycle: If Satisfied, task popped from stack. Loop may break.
    end

    CognitiveCycle-->>User: Cycle End