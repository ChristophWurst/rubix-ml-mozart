<?php

namespace Test\Vendor\Rubix\ML\Other\Traits;

use Test\Vendor\Rubix\ML\Verbose;
use Test\Vendor\Psr\Log\LoggerInterface;

/**
 * Logger Aware
 *
 * This trait fulfills the requirements of the Verbose interface and is suitable for most
 * estimators.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
trait LoggerAware
{
    /**
     * The PSR-3 logger instance.
     *
     * @var \Test\Vendor\Psr\Log\LoggerInterface|null
     */
    protected $logger;

    /**
     * Sets a logger instance on the object.
     *
     * @param \Test\Vendor\Psr\Log\LoggerInterface $logger
     */
    public function setLogger(LoggerInterface $logger) : void
    {
        $this->logger = $logger;
    }

    /**
     * Return if the logger is logging or not.
     *
     * @return \Test\Vendor\Psr\Log\LoggerInterface|null
     */
    public function logger() : ?LoggerInterface
    {
        return $this->logger;
    }
}
