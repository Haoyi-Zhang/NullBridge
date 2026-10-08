# NullBridge

Typed representation certificates for an immutable byte-buffer model.

This public companion contains the logical value domain, buffer interpreter,
and certificate checker. It validates a physical model against ordered typed
logical rows without loading a database engine, following C pointers, or
executing a third-party consumer. Database campaigns and production-witness
replays are not part of this repository.

## Run

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

`nullbridge/logical.py` defines typed values and canonical comparison;
`nullbridge/physical.py` validates and interprets owned byte buffers. A
certificate states agreement within this admitted model. It is not a claim
about every Arrow layout or database implementation. The included MIT license
applies to the original code.
