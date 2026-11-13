#!/bin/bash
# Monitor factorial jobs and cancel duplicates when one version starts running.
# Each task may have multiple job IDs (priority + scavenge + resubmits).
# When one starts RUNNING, cancel all others for the same task.
# When a scavenge job is PREEMPTED and requeued (goes back to PENDING),
# don't treat it as resolved — wait for it to actually run or for priority to start.

declare -A TASK_JOBS
# format: TASK_JOBS[task_name]="jobid1 jobid2 jobid3 ..."
# f01 already completed, f02 has only one job, f03 has only one job, f05 priority already running
TASK_JOBS[f04]="5600665 5605758"
TASK_JOBS[f11]="5600672 5605759"
TASK_JOBS[f12]="5600673 5605760"
TASK_JOBS[f13]="5600674 5605761"
TASK_JOBS[f15]="5600250 5600676"
TASK_JOBS[f16]="5600251 5600677"

resolved=()

while true; do
    all_done=true
    for task in "${!TASK_JOBS[@]}"; do
        # Skip already resolved
        if [[ " ${resolved[*]} " =~ " ${task} " ]]; then
            continue
        fi

        all_done=false
        read -ra jobs <<< "${TASK_JOBS[$task]}"
        running_job=""

        for jid in "${jobs[@]}"; do
            state=$(sacct -j "$jid" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
            if [ "$state" = "RUNNING" ]; then
                running_job="$jid"
                break
            elif [ "$state" = "COMPLETED" ]; then
                running_job="$jid"
                break
            fi
        done

        if [ -n "$running_job" ]; then
            # Cancel all other jobs for this task
            for jid in "${jobs[@]}"; do
                if [ "$jid" != "$running_job" ]; then
                    state=$(sacct -j "$jid" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
                    if [ "$state" = "PENDING" ] || [ "$state" = "RUNNING" ]; then
                        echo "$(date): $task — job $running_job active, cancelling $jid"
                        scancel "$jid"
                    fi
                fi
            done
            resolved+=("$task")
            echo "$(date): $task RESOLVED (job $running_job)"
        fi
    done

    echo "$(date): ${#resolved[@]}/${#TASK_JOBS[@]} tasks resolved"

    if $all_done; then
        echo "$(date): All tasks resolved. Exiting."
        break
    fi

    sleep 60
done
