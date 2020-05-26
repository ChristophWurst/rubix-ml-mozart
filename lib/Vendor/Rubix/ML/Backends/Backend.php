<?php

namespace Test\Vendor\Rubix\ML\Backends;

use Test\Vendor\Rubix\ML\Backends\Tasks\Task;

interface Backend
{
    /**
     * Queue up a task for backend processing.
     *
     * @param \Test\Vendor\Rubix\ML\Backends\Tasks\Task $task
     * @param callable|null $after
     */
    public function enqueue(Task $task, ?callable $after = null) : void;

    /**
     * Process the queue and return the results.
     *
     * @return mixed[]
     */
    public function process() : array;

    /**
     * Flush the queue.
     */
    public function flush() : void;
}
