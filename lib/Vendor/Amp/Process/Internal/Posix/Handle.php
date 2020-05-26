<?php

namespace Test\Vendor\Amp\Process\Internal\Posix;

use Test\Vendor\Amp\Deferred;
use Test\Vendor\Amp\Process\Internal\ProcessHandle;

/** @internal */
final class Handle extends ProcessHandle
{
    public function __construct()
    {
        $this->pidDeferred = new Deferred;
        $this->joinDeferred = new Deferred;
        $this->originalParentPid = \getmypid();
    }

    /** @var Deferred */
    public $joinDeferred;

    /** @var resource */
    public $proc;

    /** @var resource */
    public $extraDataPipe;

    /** @var string */
    public $extraDataPipeWatcher;

    /** @var string */
    public $extraDataPipeStartWatcher;

    /** @var int */
    public $originalParentPid;
}
