<?php

namespace Test\Vendor\Rubix\ML\Persisters\Serializers;

use Test\Vendor\Rubix\ML\Persistable;

interface Serializer
{
    /**
     * Serialize a persistable object and return the data.
     *
     * @param \Test\Vendor\Rubix\ML\Persistable $persistable
     * @return string
     */
    public function serialize(Persistable $persistable) : string;

    /**
     * Unserialize a persistable object and return it.
     *
     * @param string $data
     * @return \Test\Vendor\Rubix\ML\Persistable
     */
    public function unserialize(string $data) : Persistable;
}
