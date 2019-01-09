// This file is a part of Julia. License is MIT: https://julialang.org/license

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "julia.h"
#include "julia_internal.h"
#include "gc.h"
#include "threading.h"

#ifdef __cplusplus
extern "C" {
#endif

#define JULIA_ENABLE_PARTR

#ifdef JULIA_ENABLE_THREADING
#ifdef JULIA_ENABLE_PARTR

// GC functions used
extern int jl_gc_mark_queue_obj_explicit(jl_gc_mark_cache_t *gc_cache,
                                         jl_gc_mark_sp_t *sp, jl_value_t *obj);

// thread sleep threshold
extern uint64_t jl_thread_sleep_threshold;

// multiq
// ---

/* a task heap */
typedef struct taskheap_tag {
    jl_mutex_t lock;
    jl_task_t **tasks;
    int16_t ntasks, prio;
} taskheap_t;

/* multiqueue parameters */
static const int16_t heap_d = 8;
static const int heap_c = 4;

/* size of each heap */
static const int tasks_per_heap = 8192; // TODO: this should be smaller by default, but growable!

/* the multiqueue's heaps */
static taskheap_t *heaps;
static int16_t heap_p;

/* unbias state for the RNG */
static uint64_t cong_unbias;

/* for thread sleeping */
static uv_mutex_t sleep_lock;
static uv_cond_t  sleep_alarm;


/*  multiq_init()
 */
static inline void multiq_init(void)
{
    heap_p = heap_c * jl_n_threads;
    heaps = (taskheap_t *)calloc(heap_p, sizeof(taskheap_t));
    for (int16_t i = 0; i < heap_p; ++i) {
        jl_mutex_init(&heaps[i].lock);
        heaps[i].tasks = (jl_task_t **)calloc(tasks_per_heap, sizeof(jl_task_t *));
        heaps[i].ntasks = 0;
        heaps[i].prio = INT16_MAX;
    }
    unbias_cong(heap_p, &cong_unbias);
}


/*  sift_up()
 */
static inline void sift_up(taskheap_t *heap, int16_t idx)
{
    if (idx > 0) {
        int16_t parent = (idx-1)/heap_d;
        if (heap->tasks[idx]->prio < heap->tasks[parent]->prio) {
            jl_task_t *t = heap->tasks[parent];
            heap->tasks[parent] = heap->tasks[idx];
            heap->tasks[idx] = t;
            sift_up(heap, parent);
        }
    }
}


/*  sift_down()
 */
static inline void sift_down(taskheap_t *heap, int16_t idx)
{
    if (idx < heap->ntasks) {
        for (int16_t child = heap_d*idx + 1;
                child < tasks_per_heap && child <= heap_d*idx + heap_d;
                ++child) {
            if (heap->tasks[child]
                    &&  heap->tasks[child]->prio < heap->tasks[idx]->prio) {
                jl_task_t *t = heap->tasks[idx];
                heap->tasks[idx] = heap->tasks[child];
                heap->tasks[child] = t;
                sift_down(heap, child);
            }
        }
    }
}


/*  multiq_insert()
 */
static inline int multiq_insert(jl_task_t *task, int16_t priority)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    uint64_t rn;

    task->prio = priority;
    do {
        rn = cong(heap_p, cong_unbias, &ptls->rngseed);
    } while (!jl_mutex_trylock_nogc(&heaps[rn].lock));

    if (heaps[rn].ntasks >= tasks_per_heap) {
        jl_mutex_unlock_nogc(&heaps[rn].lock);
        jl_error("multiq insertion failed, increase #tasks per heap");
        return -1;
    }

    heaps[rn].tasks[heaps[rn].ntasks++] = task;
    sift_up(&heaps[rn], heaps[rn].ntasks-1);
    jl_mutex_unlock_nogc(&heaps[rn].lock);
    int16_t prio = jl_atomic_load(&heaps[rn].prio);
    if (task->prio < prio)
        jl_atomic_compare_exchange(&heaps[rn].prio, prio, task->prio);

    return 0;
}


/*  multiq_deletemin()
 */
static inline jl_task_t *multiq_deletemin(void)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    uint64_t rn1 = 0, rn2;
    int16_t i, prio1, prio2;
    jl_task_t *task;

    for (i = 0; i < heap_p; ++i) {
        rn1 = cong(heap_p, cong_unbias, &ptls->rngseed);
        rn2 = cong(heap_p, cong_unbias, &ptls->rngseed);
        prio1 = jl_atomic_load(&heaps[rn1].prio);
        prio2 = jl_atomic_load(&heaps[rn2].prio);
        if (prio1 > prio2) {
            prio1 = prio2;
            rn1 = rn2;
        }
        else if (prio1 == prio2 && prio1 == INT16_MAX)
            continue;
        if (jl_mutex_trylock_nogc(&heaps[rn1].lock)) {
            if (prio1 == heaps[rn1].prio)
                break;
            jl_mutex_unlock_nogc(&heaps[rn1].lock);
        }
    }
    if (i == heap_p)
        return NULL;

    task = heaps[rn1].tasks[0];
    heaps[rn1].tasks[0] = heaps[rn1].tasks[--heaps[rn1].ntasks];
    heaps[rn1].tasks[heaps[rn1].ntasks] = NULL;
    prio1 = INT16_MAX;
    if (heaps[rn1].ntasks > 0) {
        sift_down(&heaps[rn1], 0);
        prio1 = heaps[rn1].tasks[0]->prio;
    }
    jl_atomic_store(&heaps[rn1].prio, prio1);
    jl_mutex_unlock_nogc(&heaps[rn1].lock);

    return task;
}


// sync trees
// ---

/* arrival tree */
struct _arriver_t {
    int16_t index, next_avail;
    int16_t **tree;
};

/* reduction tree */
struct _reducer_t {
    int16_t index, next_avail;
    jl_value_t ***tree;
};


/* pool of arrival trees */
static arriver_t *arriverpool;
static int16_t num_arrivers, num_arriver_tree_nodes, next_arriver;

/* pool of reduction trees */
static reducer_t *reducerpool;
static int16_t num_reducers, num_reducer_tree_nodes, next_reducer;


/*  synctreepool_init()
 */
static inline void synctreepool_init(void)
{
    num_arriver_tree_nodes = (GRAIN_K * jl_n_threads) - 1;
    num_reducer_tree_nodes = (2 * GRAIN_K * jl_n_threads) - 1;

    /* num_arrivers = ((GRAIN_K * jl_n_threads) ^ ARRIVERS_P) + 1 */
    num_arrivers = GRAIN_K * jl_n_threads;
    for (int i = 1; i < ARRIVERS_P; ++i)
        num_arrivers = num_arrivers * num_arrivers;
    ++num_arrivers;

    num_reducers = num_arrivers * REDUCERS_FRAC;

    /* allocate */
    arriverpool = (arriver_t *)calloc(num_arrivers, sizeof (arriver_t));
    next_arriver = 0;
    for (int i = 0; i < num_arrivers; ++i) {
        arriverpool[i].index = i;
        arriverpool[i].next_avail = i + 1;
        arriverpool[i].tree = (int16_t **)
                jl_malloc_aligned(num_arriver_tree_nodes * sizeof (int16_t *), 64);
        for (int j = 0; j < num_arriver_tree_nodes; ++j)
            arriverpool[i].tree[j] = (int16_t *)jl_malloc_aligned(sizeof (int16_t), 64);
    }
    arriverpool[num_arrivers - 1].next_avail = -1;

    reducerpool = (reducer_t *)calloc(num_reducers, sizeof (reducer_t));
    next_reducer = 0;
    for (int i = 0; i < num_reducers; ++i) {
        reducerpool[i].index = i;
        reducerpool[i].next_avail = i + 1;
        reducerpool[i].tree = (jl_value_t ***)
                jl_malloc_aligned(num_reducer_tree_nodes * sizeof (jl_value_t **), 64);
        for (int j = 0; j < num_reducer_tree_nodes; ++j)
            reducerpool[i].tree[j] = (jl_value_t **)jl_malloc_aligned(sizeof (jl_value_t *), 64);
    }
    if (num_reducers > 0)
        reducerpool[num_reducers - 1].next_avail = -1;
    else
        next_reducer = -1;
}


/*  arriver_alloc()
 */
static inline arriver_t *arriver_alloc(void)
{
    int16_t candidate;
    arriver_t *arr;

    do {
        candidate = jl_atomic_load(&next_arriver);
        if (candidate == -1)
            return NULL;
        arr = &arriverpool[candidate];
    } while (!jl_atomic_bool_compare_exchange(&next_arriver,
                candidate, arr->next_avail));
    return arr;
}


/*  arriver_free()
 */
static inline void arriver_free(arriver_t *arr)
{
    for (int i = 0; i < num_arriver_tree_nodes; ++i)
        *arr->tree[i] = 0;

    jl_atomic_exchange_generic(&next_arriver, &arr->index, &arr->next_avail);
}


/*  reducer_alloc()
 */
static inline reducer_t *reducer_alloc(void)
{
    int16_t candidate;
    reducer_t *red;

    do {
        candidate = jl_atomic_load(&next_reducer);
        if (candidate == -1)
            return NULL;
        red = &reducerpool[candidate];
    } while (!jl_atomic_bool_compare_exchange(&next_reducer,
                     candidate, red->next_avail));
    return red;
}


/*  reducer_free()
 */
static inline void reducer_free(reducer_t *red)
{
    for (int i = 0; i < num_reducer_tree_nodes; ++i)
        *red->tree[i] = 0;

    jl_atomic_exchange_generic(&next_reducer, &red->index, &red->next_avail);
}


/*  last_arriver()
 */
static inline int last_arriver(arriver_t *arr, int idx)
{
    int arrived, aidx = idx + (GRAIN_K * jl_n_threads) - 1;

    while (aidx > 0) {
        --aidx;
        aidx >>= 1;
        arrived = jl_atomic_fetch_add(arr->tree[aidx], 1);
        if (!arrived) return 0;
    }

    return 1;
}


// parallel task runtime
// ---

// initialize the threading infrastructure
void jl_init_threadinginfra(void)
{
    /* initialize the synchronization trees pool and the multiqueue */
    synctreepool_init();
    multiq_init();

    /* initialize the sleep mechanism */
    uv_mutex_init(&sleep_lock);
    uv_cond_init(&sleep_alarm);
}


void JL_NORETURN jl_finish_task(jl_task_t *t, jl_value_t *resultval JL_MAYBE_UNROOTED);

// thread function: used by all except the main thread
void jl_threadfun(void *arg)
{
    jl_threadarg_t *targ = (jl_threadarg_t*)arg;

    // initialize this thread (set tid, create heap, set up root task)
    jl_init_threadtls(targ->tid);
    void *stack_lo, *stack_hi;
    jl_init_stack_limits(0, &stack_lo, &stack_hi);
    jl_init_root_task(stack_lo, stack_hi);

    // Assuming the functions called below don't contain unprotected GC
    // critical region. In general, the following part of this function
    // shouldn't call any managed code without calling `jl_gc_unsafe_enter`
    // first.
    jl_ptls_t ptls = jl_get_ptls_states();
    jl_gc_state_set(ptls, JL_GC_STATE_SAFE, 0);
    uv_barrier_wait(targ->barrier);

    // free the thread argument here
    free(targ);

    jl_current_task->exception = jl_nothing;
    jl_finish_task(jl_current_task, jl_nothing); // noreturn
}


// enqueue the specified task for execution
JL_DLLEXPORT void jl_enqueue_task(jl_task_t *task)
{
    multiq_insert(task, task->prio);

    /* wake up threads */
    if (jl_thread_sleep_threshold) {
        uv_mutex_lock(&sleep_lock);
        uv_cond_broadcast(&sleep_alarm); // TODO: make this uv_cond_signal (unless sticky)
        uv_mutex_unlock(&sleep_lock);
    }
}

// FIXME: run the next available task
static void run_next(void)
{
    jl_value_t *wait_func = jl_get_global(jl_base_module, jl_symbol("wait"));
    jl_apply(&wait_func, 1);
}
static void enqueue_task(jl_task_t *task)
{
    jl_value_t *args[2] = {
        jl_get_global(jl_base_module, jl_symbol("enq_work")),
        task };
    jl_apply(args, 2);
}

// parfor grains must synchronize/reduce as they end
static void sync_grains(jl_task_t *task)
{
    int was_last = 0;

    /* TODO kp: fix */
    /* TODO kp: cascade exception(s) if any */

    /* reduce... */
    if (task->red) {
        //task->result = reduce(task->arr, task->red, task->rfptr, task->mredfunc,
        //                      task->rargs, task->result, task->grain_num);
        jl_gc_wb(task, task->result);

        /*  if this task is last, set the result in the parent task */
        if (task->result) {
            task->parent->redresult = task->result;
            jl_gc_wb(task->parent, task->parent->redresult);
            was_last = 1;
        }
    }
    /* ... or just sync */
    else {
        if (last_arriver(task->arr, task->grain_num))
            was_last = 1;
    }

    /* the last task to finish needs to finish up the loop */
    if (was_last) {
        /* a non-parent task must wake up the parent */
        if (task->grain_num > 0)
            enqueue_task(task->parent);

        /* this is the parent task which was last; it can just end */
        if (task->red)
            reducer_free(task->red);
        arriver_free(task->arr);
    }
    else {
        /* the parent task needs to wait */
        if (task->grain_num == 0) {
            // run the next available task
            run_next();
            task->result = task->redresult;
            jl_gc_wb(task, task->result);
        }
    }
}


// all tasks except the root task exit through here
void jl_task_done_hook_partr(jl_task_t *task)
{
    /* grain tasks must synchronize */
    if (task->grain_num >= 0)
        sync_grains(task);
}


// get the next runnable task from the multiq
static jl_task_t *get_next_task(jl_value_t *getsticky)
{
    jl_task_t *task = (jl_task_t*)jl_apply(&getsticky, 1);
    if (jl_typeis(task, jl_task_type))
        return task;
    return multiq_deletemin();
}


JL_DLLEXPORT jl_task_t *jl_task_get_next(jl_value_t *getsticky)
{
    jl_ptls_t ptls = jl_get_ptls_states();
    jl_task_t *task = NULL;
    JL_GC_PUSH1(&task);

    uint64_t spin_ns, spin_start = 0;
    while (!task) {
        if (jl_thread_sleep_threshold) {
            if (spin_start == 0) {
                spin_start = uv_hrtime();
                continue;
            }
        }

        task = get_next_task(getsticky);

        if (!task) {
            if (ptls->tid == 0)
                jl_process_events(jl_global_event_loop());
            else
                jl_cpu_pause();

            if (jl_thread_sleep_threshold) {
                spin_ns = uv_hrtime() - spin_start;
                if (spin_ns > jl_thread_sleep_threshold) {
                    uv_mutex_lock(&sleep_lock);
                    task = get_next_task(getsticky);
                    if (!task) {
                        // thread 0 makes a blocking call to the event loop
                        if (ptls->tid == 0) {
                            uv_mutex_unlock(&sleep_lock);
                            jl_run_once(jl_global_event_loop());
                        }
                        // other threads just sleep
                        else {
                            uv_cond_wait(&sleep_alarm, &sleep_lock);
                            uv_mutex_unlock(&sleep_lock);
                        }
                    }
                    else {
                        uv_mutex_unlock(&sleep_lock);
                    }
                    spin_start = 0;
                }
            }
        }
    }

    JL_GC_POP();
    return task;
}


void jl_gc_mark_enqueued_tasks(jl_gc_mark_cache_t *gc_cache, jl_gc_mark_sp_t *sp)
{
    for (int16_t i = 0; i < heap_p; ++i)
        for (int16_t j = 0; j < heaps[i].ntasks; ++j)
            jl_gc_mark_queue_obj_explicit(gc_cache, sp, (jl_value_t *)heaps[i].tasks[j]);
}

#else
void jl_init_threadinginfra(void) { }
void jl_threadfun(void *arg) { abort(); }
void jl_task_done_hook_partr(jl_task_t *task) { }
#endif
#endif // JULIA_ENABLE_THREADING

#ifdef __cplusplus
}
#endif
