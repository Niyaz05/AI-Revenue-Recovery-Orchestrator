# Level 1 Hard Safety Policy & Governance

The Level 1 Safety layer is a deterministic rules engine that acts as a strict firewall between the AI / Learning policy and real-world execution.

## Hard Merchant Safety Constraints

1. **Opt-Out Compliance**:
   - If `customer.opted_out_recovery == True`, the allowed actions set is forcefully reduced to `{STOP_RECOVERY}`. No customer contact or charges can occur.

2. **Retry Attempt Cap**:
   - `attempt_count >= max_retry_attempts` (default: 3) removes `RETRY` from the permitted action set.

3. **Cooldown Window Enforcement**:
   - If `(current_time - last_attempt_time) < retry_cooldown_hours` (default: 4.0h), `RETRY` is blocked to prevent rapid card hammering and bank fraud flags.

4. **Recovery Window Limit**:
   - If `(current_time - first_failure_time) > recovery_window_hours` (default: 168h / 7 days), the recovery attempt is strictly terminated with `STOP_RECOVERY`.

5. **Human Approval for High-Value Subscriptions**:
   - Subscriptions with `amount > human_approval_threshold_amount` (default: ₹1,00,000) are routed exclusively to `ESCALATE_TO_HUMAN`.

6. **Global Merchant Retry Budget**:
   - Daily retries are tracked across the merchant account. Once `global_retries_today >= global_daily_retry_budget` (default: 100), all automated retries are disabled.

7. **Customer Daily Intervention Cap**:
   - Limits customer pings to at most 3 per day across all channels to avoid spam and brand damage.

## Mathematical Invariant Proof

Let $\mathcal{A} = \{\text{WAIT}, \text{RETRY}, \text{PAYMENT\_LINK}, \dots, \text{STOP\_RECOVERY}\}$ be the action space.
The safety policy function $S: \mathcal{X} \to 2^{\mathcal{A}}$ produces a constrained subset $\mathcal{A}_{\text{allowed}} \subseteq \mathcal{A}$.

The strategy model $M$ chooses:
$$a^* = \arg\max_{a \in \mathcal{A}_{\text{allowed}}} Q(x, a)$$

Because $a^*$ is drawn strictly from $\mathcal{A}_{\text{allowed}}$, for any forbidden action $a_{\text{forbidden}} \notin \mathcal{A}_{\text{allowed}}$, $P(a^* = a_{\text{forbidden}}) \equiv 0$.
Zero violations are mathematically guaranteed across all offline, stress, and online environments.
