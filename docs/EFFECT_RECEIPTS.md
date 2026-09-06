# External-effect evidence and recovery

Daita reserves a durable receipt before dispatching a native database change or an
admitted external action. The receipt identifies the exact capability contract,
run, call and normalized operation. A repeated operation in the same run or
occurrence is not dispatched again, including after definite non-application.

`SUCCEEDED / ADAPTER_VERIFIED` records native transaction evidence.
`SUCCEEDED / SERVER_REPORTED` means the remote tool invocation returned normally;
it does not independently verify downstream business completion. `NOT_APPLIED`
requires positive evidence of rollback or local non-dispatch. `UNCERTAIN` means
an effect may have happened. A lost response is never proof of failure.

A started receipt left by interruption is recovered as uncertain when the agent
reopens. Unresolved evidence survives conversation clearing and blocks new
foreground external effects. Read-only investigation and receipt inspection
remain available. If terminal evidence cannot be persisted, the current host
blocks further effects; reopen the agent to recover the reserved receipt.

The Python control plane provides bounded inspection and explicit recovery:

```python
from daita import EffectResolutionDecision

pending = await agent.list_effects(unresolved_only=True, limit=20, offset=0)
receipt = await agent.inspect_effect(pending[0].receipt_id)
assert receipt is not None

resolved = await agent.resolve_effect(
    receipt.receipt_id,
    expected_digest=receipt.receipt_digest,
    decision=EffectResolutionDecision.ALLOW_FUTURE_WORK,
    note="I investigated the remote system and accept possible duplication in future authorized work.",
)
```

Recovery requires the agent's foreground approval handler. The review document
shows the original receipt, exact decision, note, evidence references and affected
routine. Denial or a changed digest leaves the receipt untouched. Optional evidence
references must identify exact agent-owned terminal receipts or committed
artifacts; references and the original observation do not become new authority.

`CLOSE_WITHOUT_RETRY` disables a producing routine. `ALLOW_FUTURE_WORK` leaves it
paused and still requires ordinary explicit resume/run-now and current authority.
Both decisions preserve the original uncertain observation. Neither retries the
operation, changes its evidence basis, grants connector access, or proves that the
original action failed or succeeded. Resolution is a human control, unavailable to
model tools and scheduled instructions.

The common receipt, frozen-grant and required-effect outcome lifecycle is
implemented. Unattended production effects remain disabled pending the concrete
native/MCP action implementations and their later conformance and product gates.
