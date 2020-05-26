<?php

namespace Test\Vendor\Rubix\ML\Persisters;

use Test\Vendor\Rubix\ML\Persistable;

interface Persister
{
    /**
     * Save the persistable model.
     *
     * @param \Test\Vendor\Rubix\ML\Persistable $persistable
     */
    public function save(Persistable $persistable) : void;

    /**
     * Load the last model that was saved.
     *
     * @return \Test\Vendor\Rubix\ML\Persistable
     */
    public function load() : Persistable;
}
