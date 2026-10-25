# Model Invariants

## I1 Capacity invariant
`executed + reserved <= B`

## I2 Idempotency invariant
Duplicate requests with the same order identifier must not consume capacity twice.

## I3 Authoritative-release invariant
ACK or timeout alone never releases reserved capacity.

## I4 Residual-reservation invariant
Following a partial fill, the unfilled quantity remains reserved until a terminal fill, cancellation, rejection, or authoritative reconciliation.
