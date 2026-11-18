//! The agent estimates the electricity cost of an offer before accepting it.
//! Divide one million tokens by hourly throughput to get the hours of generation.
//! Multiplying that time by power in kilowatts and the electricity tariff gives
//! the cost per million tokens. An offer can proceed without a throughput
//! estimate so the first job can provide one.

use publicompute_proto::{common::Money, node::PayoutRate};

#[derive(Debug, Clone)]
pub struct HostCosts {
    /// Power draw under load, in watts, with the configured power limit applied.
    pub power_watts: f64,
    pub electricity: Money,
}

/// Estimate electricity cost in micro-units per million generated tokens.
pub fn cost_micros_per_million(costs: &HostCosts, tokens_per_sec: f64) -> Option<i64> {
    if tokens_per_sec <= 0.0 || costs.power_watts <= 0.0 {
        return None;
    }
    let price_per_kwh = costs.electricity.micros as f64; // Tariffs are stored in micro-units per kilowatt-hour.
    let seconds_per_million = 1e6 / tokens_per_sec;
    let kwh = costs.power_watts / 1000.0 * seconds_per_million / 3600.0;
    Some((kwh * price_per_kwh).round() as i64)
}

/// Accept an offer if its estimated cost does not exceed the reward.
/// If the currencies differ, log the mismatch and accept the offer.
pub fn is_profitable(payout: Option<&PayoutRate>, costs: Option<&HostCosts>, tokens_per_sec: Option<f64>) -> bool {
    let (Some(payout), Some(costs), Some(tps)) = (payout, costs, tokens_per_sec) else {
        return true;
    };
    let Some(out) = payout.per_million_output_tokens.as_ref() else {
        return true;
    };
    if out.currency != costs.electricity.currency {
        return true;
    }
    match cost_micros_per_million(costs, tps) {
        Some(cost) => out.micros >= cost,
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eur(m: i64) -> Money {
        Money { micros: m, currency: "EUR".into() }
    }

    #[test]
    fn cost_example() {
        // A million tokens at 50 per second take about 5.56 hours. A draw of 300 W
        // at €0.30 per kWh costs about €0.50 for that time.
        let c = HostCosts { power_watts: 300.0, electricity: eur(300_000) };
        let cost = cost_micros_per_million(&c, 50.0).unwrap();
        assert!((cost - 500_000).abs() <= 1, "costo = {cost}");
    }

    #[test]
    fn profitability() {
        let c = HostCosts { power_watts: 300.0, electricity: eur(300_000) };
        let pay = |m| PayoutRate { per_million_input_tokens: None, per_million_output_tokens: Some(eur(m)) };
        assert!(is_profitable(Some(&pay(600_000)), Some(&c), Some(50.0)));
        assert!(!is_profitable(Some(&pay(400_000)), Some(&c), Some(50.0)));
        assert!(is_profitable(Some(&pay(1)), Some(&c), None));
        assert!(is_profitable(Some(&pay(1)), None, Some(50.0)));
    }

    /// A missing reward leaves the cost check undecided. An explicit zero reward
    /// fails when the host reports a positive cost.
    #[test]
    fn accepts_probe_with_catalog_compensation() {
        let c = HostCosts { power_watts: 300.0, electricity: eur(300_000) };
        let catalogo = PayoutRate { per_million_input_tokens: None, per_million_output_tokens: Some(eur(2_000_000)) };
        assert!(is_profitable(Some(&catalogo), Some(&c), Some(50.0)), "accept the catalog rate");
        let zero = PayoutRate { per_million_input_tokens: None, per_million_output_tokens: Some(eur(0)) };
        assert!(!is_profitable(Some(&zero), Some(&c), Some(50.0)), "reject an explicit zero rate");
        assert!(is_profitable(None, Some(&c), Some(50.0)));
    }
}
