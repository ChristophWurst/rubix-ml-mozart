<?php

namespace Test\Vendor\Rubix\ML;

interface Wrapper extends Estimator
{
    /**
     * Return the base estimator instance.
     *
     * @return \Test\Vendor\Rubix\ML\Estimator
     */
    public function base() : Estimator;

    /**
     * Allow methods to be called on the estimator from the wrapper.
     *
     * @param string $name
     * @param mixed[] $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments);
}
