<?php

namespace Test\Vendor\Rubix\ML;

use Test\Vendor\Rubix\ML\Backends\Backend;

interface Parallel
{
    /**
     * Set the parallel processing backend.
     *
     * @param \Test\Vendor\Rubix\ML\Backends\Backend $backend
     */
    public function setBackend(Backend $backend) : void;
}
