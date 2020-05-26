<?php

namespace Test\Vendor\Amp\Parallel\Sync;

use Test\Vendor\Amp\Promise;

/**
 * Interface for sending messages between execution contexts.
 */
interface Channel
{
    /**
     * @return \Test\Vendor\Amp\Promise<mixed>
     *
     * @throws \Test\Vendor\Amp\Parallel\Context\StatusError Thrown if the context has not been started.
     * @throws \Test\Vendor\Amp\Parallel\Sync\SynchronizationError If the context has not been started or the context
     *     unexpectedly ends.
     * @throws \Test\Vendor\Amp\Parallel\Sync\ChannelException If receiving from the channel fails.
     * @throws \Test\Vendor\Amp\Parallel\Sync\SerializationException If unserializing the data fails.
     */
    public function receive(): Promise;

    /**
     * @param mixed $data
     *
     * @return \Test\Vendor\Amp\Promise<int> Resolves with the number of bytes sent on the channel.
     *
     * @throws \Test\Vendor\Amp\Parallel\Context\StatusError Thrown if the context has not been started.
     * @throws \Test\Vendor\Amp\Parallel\Sync\SynchronizationError If the context has not been started or the context
     *     unexpectedly ends.
     * @throws \Test\Vendor\Amp\Parallel\Sync\ChannelException If sending on the channel fails.
     * @throws \Error If an ExitResult object is given.
     * @throws \Test\Vendor\Amp\Parallel\Sync\SerializationException If serializing the data fails.
     */
    public function send($data): Promise;
}
