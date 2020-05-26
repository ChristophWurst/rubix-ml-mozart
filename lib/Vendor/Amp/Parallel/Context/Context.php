<?php

namespace Test\Vendor\Amp\Parallel\Context;

use Test\Vendor\Amp\Parallel\Sync\Channel;
use Test\Vendor\Amp\Promise;

interface Context extends Channel
{
    /**
     * @return bool
     */
    public function isRunning(): bool;

    /**
     * Starts the execution context.
     *
     * @return Promise<null> Resolved once the context has started.
     */
    public function start(): Promise;

    /**
     * Immediately kills the context.
     */
    public function kill();

    /**
     * @return \Test\Vendor\Amp\Promise<mixed> Resolves with the returned from the context.
     *
     * @throws \Test\Vendor\Amp\Parallel\Context\ContextException If the context dies unexpectedly.
     * @throws \Test\Vendor\Amp\Parallel\Sync\PanicError If the context throws an uncaught exception.
     */
    public function join(): Promise;
}
