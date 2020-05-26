<?php

namespace Test\Vendor\Amp\Sync;

use Test\Vendor\Amp\CallableMaker;
use Test\Vendor\Amp\Deferred;
use Test\Vendor\Amp\Promise;
use Test\Vendor\Amp\Success;

class LocalMutex implements Mutex
{
    use CallableMaker; // kept for BC only

    /** @var bool */
    private $locked = false;

    /** @var Deferred[] */
    private $queue = [];

    /** {@inheritdoc} */
    public function acquire(): Promise
    {
        if (!$this->locked) {
            $this->locked = true;
            return new Success(new Lock(0, \Closure::fromCallable([$this, 'release'])));
        }

        $this->queue[] = $deferred = new Deferred;
        return $deferred->promise();
    }

    private function release(): void
    {
        if (!empty($this->queue)) {
            $deferred = \array_shift($this->queue);
            $deferred->resolve(new Lock(0, \Closure::fromCallable([$this, 'release'])));
            return;
        }

        $this->locked = false;
    }
}
