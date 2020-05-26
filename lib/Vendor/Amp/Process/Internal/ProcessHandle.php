<?php

namespace Test\Vendor\Amp\Process\Internal;

use Test\Vendor\Amp\Deferred;
use Test\Vendor\Amp\Process\ProcessInputStream;
use Test\Vendor\Amp\Process\ProcessOutputStream;
use Test\Vendor\Amp\Struct;

abstract class ProcessHandle
{
    use Struct;

    /** @var ProcessOutputStream */
    public $stdin;

    /** @var ProcessInputStream */
    public $stdout;

    /** @var ProcessInputStream */
    public $stderr;

    /** @var Deferred */
    public $pidDeferred;

    /** @var int */
    public $status = ProcessStatus::STARTING;
}
