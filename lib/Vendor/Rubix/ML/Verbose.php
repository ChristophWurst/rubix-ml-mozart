<?php

namespace Test\Vendor\Rubix\ML;

use Test\Vendor\Psr\Log\LoggerInterface;
use Test\Vendor\Psr\Log\LoggerAwareInterface;

interface Verbose extends LoggerAwareInterface
{
    /**
     * Return the logger or null if not set.
     *
     * @return \Test\Vendor\Psr\Log\LoggerInterface|null
     */
    public function logger() : ?LoggerInterface;
}
