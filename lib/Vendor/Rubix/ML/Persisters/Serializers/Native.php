<?php

namespace Test\Vendor\Rubix\ML\Persisters\Serializers;

use Test\Vendor\Rubix\ML\Persistable;

/**
 * Native
 *
 * The native PHP plain text serialization format.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Native implements Serializer
{
    /**
     * Serialize a persistable object and return the data.
     *
     * @param \Test\Vendor\Rubix\ML\Persistable $persistable
     * @return string
     */
    public function serialize(Persistable $persistable) : string
    {
        return serialize($persistable);
    }

    /**
     * Unserialize a persistable object and return it.
     *
     * @param string $data
     * @return \Test\Vendor\Rubix\ML\Persistable
     */
    public function unserialize(string $data) : Persistable
    {
        return unserialize($data);
    }
}
