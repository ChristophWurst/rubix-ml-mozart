<?php

namespace Test\Vendor\Amp\Parallel\Worker\Internal;

use Test\Vendor\Amp\Failure;
use Test\Vendor\Amp\Parallel\Worker\Task;
use Test\Vendor\Amp\Promise;
use Test\Vendor\Amp\Success;

/** @internal */
final class TaskSuccess extends TaskResult
{
    /** @var mixed Result of task. */
    private $result;

    public function __construct(string $id, $result)
    {
        parent::__construct($id);
        $this->result = $result;
    }

    public function promise(): Promise
    {
        if ($this->result instanceof \__PHP_Incomplete_Class) {
            return new Failure(new \Error(\sprintf(
                "Class instances returned from %s::run() must be autoloadable by the Composer autoloader",
                Task::class
            )));
        }

        return new Success($this->result);
    }
}
