<?php

namespace Test\Vendor\Amp\Loop\Internal;

use Test\Vendor\Amp\Loop\Watcher;
use Test\Vendor\Amp\Struct;

/**
 * @internal
 */
final class TimerQueueEntry
{
    use Struct;

    /** @var Watcher */
    public $watcher;

    /** @var int */
    public $expiration;

    /**
     * @param Watcher $watcher
     * @param int     $expiration
     */
    public function __construct(Watcher $watcher, int $expiration)
    {
        $this->watcher = $watcher;
        $this->expiration = $expiration;
    }
}
