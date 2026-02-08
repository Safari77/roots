// Copyright (c) 2015, Mikhail Vorotilov
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use super::super::FloatType;
use super::super::Roots;

// Helper: Integer power for generic FloatType. Used for scaling thresholds.
fn pow_n<F: FloatType>(base: F, mut exp: u32) -> F {
    let mut res = F::from(1i16);
    let mut b = base;
    while exp > 0 {
        if exp % 2 == 1 {
            res = res * b;
        }
        b = b * b;
        exp /= 2;
    }
    res
}

// Helper: Veltkamp splitting
fn veltkamp_split<F: FloatType>(x: F) -> (F, F) {
    let is_f32 = std::mem::size_of::<F>() == 4;
    let factor = if is_f32 {
        F::from(4097i16)
    } else {
        let two = F::from(2i16);
        // 2^27 + 1.
        let p27 = pow_n(two, 27);
        p27 + F::from(1i16)
    };

    let gamma = factor * x;
    let delta = x - gamma;
    let x_hi = gamma + delta;
    let x_lo = x - x_hi;
    (x_hi, x_lo)
}

// Helper: Exact multiplication (Dekker's Product)
fn exact_mult<F: FloatType>(x: F, y: F, product: F) -> F {
    let (x_hi, x_lo) = veltkamp_split(x);
    let (y_hi, y_lo) = veltkamp_split(y);
    let t1 = -product + x_hi * y_hi;
    let t2 = t1 + x_hi * y_lo;
    let t3 = t2 + x_lo * y_hi;
    t3 + x_lo * y_lo
}

// Helper: Kahan discriminant
fn kahan_discriminant_exact<F: FloatType>(a: F, b: F, c: F) -> F {
    let _4 = F::from(4i16);
    let _3 = F::from(3i16);

    let prod_bb = b * b;
    let prod_4ac = _4 * a * c;

    let discriminant = prod_bb - prod_4ac;

    let sum_magnitudes = prod_bb.abs() + prod_4ac.abs();

    if _3 * discriminant.abs() >= sum_magnitudes {
        discriminant
    } else {
        let err_bb = exact_mult(b, b, prod_bb);
        let err_4ac = exact_mult(_4 * a, c, prod_4ac);
        (prod_bb - prod_4ac) + (err_bb - err_4ac)
    }
}

/// Solves a quadratic equation a2*x^2 + a1*x + a0 = 0.
///
/// In case two roots are present, the first returned root is less than the second one.
///
/// # Examples
///
/// ```
/// use roots::Roots;
/// use roots::find_roots_quadratic;
///
/// let no_roots = find_roots_quadratic(1f32, 0f32, 1f32);
/// // Returns Roots::No([]) as 'x^2 + 1 = 0' has no roots
///
/// let one_root = find_roots_quadratic(1f64, 0f64, 0f64);
/// // Returns Roots::One([0f64]) as 'x^2 = 0' has one root 0
///
/// let two_roots = find_roots_quadratic(1f32, 0f32, -1f32);
/// // Returns Roots::Two([-1f32,1f32]) as 'x^2 - 1 = 0' has roots -1 and 1
/// ```
pub fn find_roots_quadratic<F: FloatType>(a2: F, a1: F, a0: F) -> Roots<F> {
    // 1. Handle NaN
    if a2 != a2 || a1 != a1 || a0 != a0 {
        return Roots::No([]);
    }

    // 2. Handle Infinity
    let zero = F::from(0i16);
    let one = F::from(1i16);
    let inf = one / zero;
    let neg_inf = -inf;
    let is_inf = |x: F| x == inf || x == neg_inf;

    if is_inf(a2) || is_inf(a1) || is_inf(a0) {
        return Roots::No([]);
    }

    // 3. Handle Degenerate Cases
    if a2 == zero {
        return super::linear::find_roots_linear(a1, a0);
    }
    if a0 == zero {
        let root2 = -a1 / a2;
        if root2 == zero {
            return Roots::One([zero]);
        }
        return if root2 < zero {
            Roots::Two([root2, zero])
        } else {
            Roots::Two([zero, root2])
        };
    }
    if a1 == zero {
        // a0 and a2 must have opposite signs for real roots
        if (a0 > zero) == (a2 > zero) {
            return Roots::No([]);
        } else {
            // Compute sqrt(|a0|/|a2|) as sqrt(|a0|)/sqrt(|a2|) to avoid overflow
            let r = a0.abs().sqrt() / a2.abs().sqrt();
            return Roots::Two([-r, r]);
        }
    }

    // 4. Scaling (Overflow/Underflow Protection)
    let abs_a2 = a2.abs();
    let abs_a1 = a1.abs();
    let abs_a0 = a0.abs();

    let mut max_abs = abs_a2;
    if abs_a1 > max_abs {
        max_abs = abs_a1;
    }
    if abs_a0 > max_abs {
        max_abs = abs_a0;
    }

    let is_f32 = std::mem::size_of::<F>() == 4;
    let two = F::from(2i16);

    // Thresholds (Powers of 2)
    let (low_threshold, high_threshold, scale_up) = if is_f32 {
        (one / pow_n(two, 60), pow_n(two, 60), pow_n(two, 64))
    } else {
        // Construct 2^600 safely
        let p200 = pow_n(two, 200);
        let p600 = p200 * p200 * p200;
        (one / pow_n(two, 500), pow_n(two, 500), p600)
    };

    let (sc_a2, sc_a1, sc_a0) = if max_abs < low_threshold && max_abs > zero {
        (a2 * scale_up, a1 * scale_up, a0 * scale_up)
    } else if max_abs > high_threshold {
        let prod_bb = a1 * a1;
        let _4 = F::from(4i16);
        let prod_ac = _4 * a2 * a0;

        if is_inf(prod_bb) || is_inf(prod_ac) {
            let scale_down = one / max_abs;
            (a2 * scale_down, a1 * scale_down, a0 * scale_down)
        } else {
            (a2, a1, a0)
        }
    } else {
        (a2, a1, a0)
    };

    // 5. Accurate Discriminant
    let discriminant = kahan_discriminant_exact(sc_a2, sc_a1, sc_a0);

    // 6. Tolerance Check
    let epsilon = if is_f32 { one / pow_n(two, 23) } else { one / pow_n(two, 52) };

    let b2 = sc_a1 * sc_a1;
    let tolerance = epsilon * b2;

    if discriminant < -tolerance {
        Roots::No([])
    } else {
        let _2 = F::from(2i16);
        let safe_d = if discriminant < zero { zero } else { discriminant };

        if safe_d == zero {
            Roots::One([-sc_a1 / (_2 * sc_a2)])
        } else {
            let sq = safe_d.sqrt();
            let q = if sc_a1 >= zero {
                -(sc_a1 + sq) / _2
            } else {
                -(sc_a1 - sq) / _2
            };

            let x1 = q / sc_a2;
            let x2 = sc_a0 / q;

            if x1 < x2 {
                Roots::Two([x1, x2])
            } else {
                Roots::Two([x2, x1])
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::super::super::*;
    use super::*;

    fn assert_roots_approx(actual: Roots<f64>, expected: &[f64]) {
        match (&actual, expected.len()) {
            (Roots::No(_), 0) => {}
            (Roots::One([r1]), 1) => {
                let e = expected[0];
                if r1.is_infinite() && e.is_infinite() {
                    assert_eq!(r1.signum(), e.signum(), "Infinite sign mismatch");
                } else {
                    let err = (r1 - e).abs();
                    // Tolerance 1e-11 for low-precision test data
                    let tolerance = 1e-11 * e.abs().max(1.0);
                    assert!(err <= tolerance, "One Root mismatch. Got {}, Expected {}", r1, e);
                }
            }
            (Roots::Two([r1, r2]), 2) => {
                let mut exp = expected.to_vec();
                exp.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let check = |got: f64, want: f64, label: &str| {
                    if got.is_infinite() && want.is_infinite() {
                        assert_eq!(got.signum(), want.signum(), "{} Infinite Sign mismatch", label);
                    } else {
                        let err = (got - want).abs();
                        let tol = 1e-11 * want.abs().max(1.0);
                        assert!(err <= tol, "{} mismatch. Got {}, Expected {}", label, got, want);
                    }
                };

                check(*r1, exp[0], "Root 1");
                check(*r2, exp[1], "Root 2");
            }
            _ => panic!("Structure mismatch. Got {:?}, Expected {:?}", actual, expected),
        }
    }

    #[test]
    fn test_find_roots_quadratic_small_a2() {
        assert_eq!(
            find_roots_quadratic(1e-20f32, -1f32, -1e-30f32),
            Roots::Two([-1e-30f32, 1e20f32])
        );
        assert_eq!(
            find_roots_quadratic(-1e-20f32, 1f32, 1e-30f32),
            Roots::Two([-1e-30f32, 1e20f32])
        );
        assert_eq!(find_roots_quadratic(1e-20f32, -1f32, 1f32), Roots::Two([1f32, 1e20f32]));
        assert_eq!(find_roots_quadratic(-1e-20f32, 1f32, 1f32), Roots::Two([-1f32, 1e20f32]));
        assert_eq!(find_roots_quadratic(-1e-20f32, 1f32, -1f32), Roots::Two([1f32, 1e20f32]));
    }

    #[test]
    fn test_find_roots_quadratic_big_a1() {
        assert_eq!(find_roots_quadratic(1f32, -1e15f32, -1f32), Roots::Two([-1e-15f32, 1e15f32]));
        assert_eq!(find_roots_quadratic(-1f32, 1e15f32, 1f32), Roots::Two([-1e-15f32, 1e15f32]));
    }

    #[test]
    fn test_find_roots_quadratic() {
        assert_eq!(find_roots_quadratic(0f32, 0f32, 0f32), Roots::One([0f32]));
        assert_eq!(find_roots_quadratic(1f32, 0f32, 1f32), Roots::No([]));
        assert_eq!(find_roots_quadratic(1f64, 0f64, -1f64), Roots::Two([-1f64, 1f64]));
    }

    #[test]
    fn test_no_solution_nan_inf() {
        let nan = f64::NAN;
        let inf = f64::INFINITY;
        let cases = [
            (1.0, 2.0, nan),
            (1.0, nan, 3.0),
            (1.0, nan, nan),
            (nan, 2.0, 3.0),
            (nan, 2.0, nan),
            (nan, nan, 3.0),
            (nan, nan, nan),
            (0.0, 0.0, 1.0),
            (1.0, 2.0, inf),
            (1.0, inf, 3.0),
            (1.0, inf, inf),
            (inf, 2.0, 3.0),
            (inf, 2.0, inf),
            (inf, inf, 3.0),
            (inf, inf, inf),
        ];
        for (a, b, c) in cases {
            assert_eq!(
                find_roots_quadratic(a, b, c),
                Roots::No([]),
                "Failed on NaN/Inf case: ({},{},{})",
                a,
                b,
                c
            );
        }
    }

    #[test]
    fn test_degenerate_cases() {
        let pow2_600 = 2.0f64.powi(600);
        let pow2_neg600 = 2.0f64.powi(-600);

        assert_roots_approx(find_roots_quadratic(0.0, 1.0, 0.0), &[0.0]);
        assert_roots_approx(find_roots_quadratic(0.0, 1.0, 2.0), &[-2.0]);
        assert_roots_approx(find_roots_quadratic(0.0, pow2_600, -pow2_600), &[1.0]);

        // This case returns -Inf in linear solver
        assert_roots_approx(find_roots_quadratic(0.0, pow2_neg600, pow2_600), &[f64::NEG_INFINITY]);

        assert_roots_approx(find_roots_quadratic(3.0, 0.0, 0.0), &[0.0]); // One([0]) or Two([0,0]) accepted

        let r = (1.5f64).sqrt();
        assert_roots_approx(find_roots_quadratic(2.0, 0.0, -3.0), &[-r, r]);

        let pow2_700 = 2.0f64.powi(700);
        assert_roots_approx(find_roots_quadratic(pow2_600, pow2_700, 0.0), &[-2.0f64.powi(100), 0.0]);
        assert_roots_approx(find_roots_quadratic(pow2_neg600, pow2_700, 0.0), &[f64::NEG_INFINITY, 0.0]);
    }

    #[test]
    fn test_two_solutions_comprehensive() {
        let pow2_neg52 = 2.0f64.powi(-52);
        let pow2_neg53 = 2.0f64.powi(-53);
        let pow2_neg51 = 2.0f64.powi(-51);

        let pow2_neg511 = 2.0f64.powi(-511);
        let pow2_neg563 = 2.0f64.powi(-563);
        let pow2_neg1024 = 2.0f64.powi(-1023) * 0.5;

        let pow2_27 = 2.0f64.powi(27);
        let pow2_600 = 2.0f64.powi(600);
        let pow2_neg600 = 2.0f64.powi(-600);
        let pow2_800 = 2.0f64.powi(800);
        let pow2_500 = 2.0f64.powi(500);
        let pow2_26 = 2.0f64.powi(26);
        let pow2_neg1073 = 2.0f64.powi(-1023) * 2.0f64.powi(-50);
        let pow2_1022 = 2.0f64.powi(1022);
        let pow2_neg1026 = 2.0f64.powi(-1023) * 2.0f64.powi(-3);

        let cases = vec![
            (1.0, -1.0, -1.0, -0.6180339887498948, 1.618033988749895),
            (1.0, 1.0 + pow2_neg52, 0.25 + pow2_neg53, (-1.0 - pow2_neg51) / 2.0, -0.5),
            (
                1.0,
                pow2_neg511 + pow2_neg563,
                pow2_neg1024,
                -7.458340888372987e-155,
                -7.458340574027429e-155,
            ),
            (1.0, pow2_27, 0.75, -134217728.0, -5.587935447692871e-09),
            (1.0, -1e9, 1.0, 1e-09, 1000000000.0),
            (
                1.3407807929942596e154,
                -1.3407807929942596e154,
                -1.3407807929942596e154,
                -0.6180339887498948,
                1.618033988749895,
            ),
            // The case that previously failed due to scaling underflow:
            (pow2_600, 0.5, -pow2_neg600, -3.086568504549085e-181, 1.8816085719976428e-181),
            (pow2_600, 0.5, -pow2_600, -1.0, 1.0),
            (8.0, pow2_800, -pow2_500, -8.335018041099818e+239, 4.909093465297727e-91),
            (1.0, pow2_26, -0.125, -67108864.0, 1.862645149230957e-09),
            (
                pow2_neg1073,
                -pow2_neg1073,
                -pow2_neg1073,
                -0.6180339887498948,
                1.618033988749895,
            ),
            (
                pow2_600,
                -pow2_neg600,
                -pow2_neg600,
                -2.409919865102884e-181,
                2.409919865102884e-181,
            ),
            (-158114166017.0, 316227766017.0, -158113600000.0, 0.99999642020057874, 1.0),
            (
                -312499999999.0,
                707106781186.0,
                -400000000000.0,
                1.131369396027,
                1.131372303775,
            ),
            (-67.0, 134.0, -65.0, 0.82722631488372798, 1.17277368511627202),
            (
                0.247260273973,
                0.994520547945,
                -0.138627953316,
                -4.157030027041105,
                0.1348693622211607,
            ),
            (1.0, -2300000.0, 2.0e11, 90518.994979145, 2209481.005020854),
            (
                1.5 * pow2_neg1026,
                0.0,
                -pow2_1022,
                -1.4678102981723264e308,
                1.4678102981723264e308,
            ),
        ];

        for (a, b, c, x1, x2) in cases {
            assert_roots_approx(find_roots_quadratic(a, b, c), &[x1, x2]);
        }
    }
}
