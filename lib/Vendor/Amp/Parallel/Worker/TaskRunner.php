<?php

namespace Test\Vendor\Amp\Parallel\Worker;

use Test\Vendor\Amp\Coroutine;
use Test\Vendor\Amp\Parallel\Sync\Channel;
use Test\Vendor\Amp\Parallel\Sync\SerializationException;
use Test\Vendor\Amp\Promise;
use function Test\Vendor\Amp\call;

final class TaskRunner
{
    /** @var Channel */
    private $channel;

    /** @var Environment */
    private $environment;

    public function __construct(Channel $channel, Environment $environment)
    {
        $this->channel = $channel;
        $this->environment = $environment;
    }

    /**
     * Runs the task runner, receiving tasks from the parent and sending the result of those tasks.
     *
     * @return \Test\Vendor\Amp\Promise
     */
    public function run(): Promise
    {
        return new Coroutine($this->execute());
    }

    /**
     * @coroutine
     *
     * @return \Generator
     */
    private function execute(): \Generator
    {
        $job = yield $this->channel->receive();

        while ($job instanceof Internal\Job) {
            try {
                $result = yield call([$job->getTask(), "run"], $this->environment);
                $result = new Internal\TaskSuccess($job->getId(), $result);
            } catch (\Throwable $exception) {
                $result = new Internal\TaskFailure($job->getId(), $exception);
            }

            $job = null; // Free memory from last job.

            try {
                yield $this->channel->send($result);
            } catch (SerializationException $exception) {
                // Could not serialize task result.
                yield $this->channel->send(new Internal\TaskFailure($result->getId(), $exception));
            }

            $result = null; // Free memory from last result.

            $job = yield $this->channel->receive();
        }

        return $job;
    }
}
