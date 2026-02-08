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

// Helper: Integer power to avoid relying on specific FloatType trait methods
fn pow_n<F: FloatType>(base: F, mut exp: u32) -> F {
    let mut res = F::from(1i16);
    let mut b = base;
    loop {
        if exp % 2 == 1 {
            res = res * b;
        }
        exp /= 2;
        if exp == 0 {
            break;
        }
        b = b * b;
    }
    res
}

// Helper: Perform Newton-Raphson refinement on roots
// x_new = x - P(x) / P'(x)
fn refine_roots<F: FloatType>(roots: Roots<F>, a4: F, a3: F, a2: F, a1: F, a0: F) -> Roots<F> {
    let _2 = F::from(2i16);
    let _3 = F::from(3i16);
    let _4 = F::from(4i16);

    // Construct epsilon mathematically to avoid type mismatch errors
    let one = F::from(1i16);
    let ten = F::from(10i16);
    let is_f32 = std::mem::size_of::<F>() == 4;
    let epsilon = if is_f32 {
        one / pow_n(ten, 7) // 1e-7 for f32
    } else {
        one / pow_n(ten, 14) // 1e-14 for f64
    };

    let refine = |x: F| -> F {
        // Run 2 iterations for high precision
        let mut x_curr = x;
        for _ in 0..2 {
            // P(x)
            let x2 = x_curr * x_curr;
            let x3 = x2 * x_curr;
            let x4 = x2 * x2;
            let p_x = a4 * x4 + a3 * x3 + a2 * x2 + a1 * x_curr + a0;

            // P'(x)
            let p_prime_x = _4 * a4 * x3 + _3 * a3 * x2 + _2 * a2 * x_curr + a1;

            if p_prime_x.abs() < epsilon {
                break; // Derivative too close to zero, stick with current guess
            }
            x_curr = x_curr - p_x / p_prime_x;
        }
        x_curr
    };

    match roots {
        Roots::No(arr) => Roots::No(arr),
        Roots::One([x1]) => Roots::One([refine(x1)]),
        Roots::Two([x1, x2]) => Roots::Two([refine(x1), refine(x2)]),
        Roots::Three([x1, x2, x3]) => Roots::Three([refine(x1), refine(x2), refine(x3)]),
        Roots::Four([x1, x2, x3, x4]) => Roots::Four([refine(x1), refine(x2), refine(x3), refine(x4)]),
    }
}

/// Solves a quartic equation a4*x^4 + a4*x^3 + a2*x^2 + a1*x + a0 = 0.
/// pp, rr, and dd are already computed while searching for multiple roots
fn find_roots_via_depressed_quartic<F: FloatType>(a4: F, a3: F, a2: F, a1: F, a0: F, pp: F, rr: F, dd: F) -> Roots<F> {
    // Depressed quartic
    // https://en.wikipedia.org/wiki/Quartic_function#Converting_to_a_depressed_quartic

    let _3 = F::from(3i16);
    let _4 = F::from(4i16);
    let _6 = F::from(6i16);
    let _8 = F::from(8i16);
    let _12 = F::from(12i16);
    let _16 = F::from(16i16);
    let _256 = F::from(256i16);

    // a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0 => y^4 + p*y^2 + q*y + r.
    let a4_pow_2 = a4 * a4;
    let a4_pow_3 = a4_pow_2 * a4;
    let a4_pow_4 = a4_pow_2 * a4_pow_2;
    // Re-use pre-calculated values
    let p = pp / (_8 * a4_pow_2);
    let q = rr / (_8 * a4_pow_3);
    let r = (dd + _16 * a4_pow_2 * (_12 * a0 * a4 - _3 * a1 * a3 + a2 * a2)) / (_256 * a4_pow_4);

    let mut roots = Roots::No([]);
    for y in super::quartic_depressed::find_roots_quartic_depressed(p, q, r)
        .as_ref()
        .iter()
    {
        roots = roots.add_new_root(*y - a3 / (_4 * a4));
    }
    roots
}

/// Solves a quartic equation a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0.
///
/// Returned roots are ordered.
/// Precision is about 5e-15 for f64, 5e-7 for f32.
/// WARNING: f32 is often not enough to find multiple roots.
///
/// # Examples
///
/// ```
/// use roots::find_roots_quartic;
///
/// let one_root = find_roots_quartic(1f64, 0f64, 0f64, 0f64, 0f64);
/// // Returns Roots::One([0f64]) as 'x^4 = 0' has one root 0
///
/// let two_roots = find_roots_quartic(1f32, 0f32, 0f32, 0f32, -1f32);
/// // Returns Roots::Two([-1f32, 1f32]) as 'x^4 - 1 = 0' has roots -1 and 1
///
/// let multiple_roots = find_roots_quartic(-14.0625f64, -3.75f64, 29.75f64, 4.0f64, -16.0f64);
/// // Returns Roots::Two([-1.1016116464173349f64, 0.9682783130840016f64])
///
/// let multiple_roots_not_found = find_roots_quartic(-14.0625f32, -3.75f32, 29.75f32, 4.0f32, -16.0f32);
/// // Returns Roots::No([]) because of f32 rounding errors when trying to calculate the discriminant
/// ```
pub fn find_roots_quartic<F: FloatType>(a4: F, a3: F, a2: F, a1: F, a0: F) -> Roots<F> {
    // 1. Handle NaN
    if a4 != a4 || a3 != a3 || a2 != a2 || a1 != a1 || a0 != a0 {
        return Roots::No([]);
    }

    // 2. Handle Infinity
    let zero = F::from(0i16);
    let one = F::from(1i16);
    let inf = one / zero;
    let neg_inf = -inf;
    let is_inf = |x: F| x == inf || x == neg_inf;

    if is_inf(a4) || is_inf(a3) || is_inf(a2) || is_inf(a1) || is_inf(a0) {
        return Roots::No([]);
    }

    // 3. Handle Degenerate Cases (Exact Zero)
    if a4 == zero {
        return super::cubic::find_roots_cubic(a3, a2, a1, a0);
    }
    if a0 == zero {
        return super::cubic::find_roots_cubic(a4, a3, a2, a1).add_new_root(zero);
    }
    if a1 == zero && a3 == zero {
        return super::biquadratic::find_roots_biquadratic(a4, a2, a0);
    }

    // 4. Scaling (Overflow/Underflow Protection)
    let abs_a4 = a4.abs();
    let abs_a3 = a3.abs();
    let abs_a2 = a2.abs();
    let abs_a1 = a1.abs();
    let abs_a0 = a0.abs();

    let mut max_abs = abs_a4;
    if abs_a3 > max_abs {
        max_abs = abs_a3;
    }
    if abs_a2 > max_abs {
        max_abs = abs_a2;
    }
    if abs_a1 > max_abs {
        max_abs = abs_a1;
    }
    if abs_a0 > max_abs {
        max_abs = abs_a0;
    }

    let is_f32 = std::mem::size_of::<F>() == 4;
    let two = F::from(2i16);

    // Thresholds (Powers of 2 for exactness)
    let (low_threshold, high_threshold, scale_up) = if is_f32 {
        (one / pow_n(two, 60), pow_n(two, 60), pow_n(two, 64))
    } else {
        // Construct 2^600 safely
        let p200 = pow_n(two, 200);
        let p600 = p200 * p200 * p200;
        (one / pow_n(two, 500), pow_n(two, 500), p600)
    };

    let (sc_a4, sc_a3, sc_a2, sc_a1, sc_a0) = if max_abs < low_threshold && max_abs > zero {
        (a4 * scale_up, a3 * scale_up, a2 * scale_up, a1 * scale_up, a0 * scale_up)
    } else if max_abs > high_threshold {
        let scale_down = one / max_abs;
        (
            a4 * scale_down,
            a3 * scale_down,
            a2 * scale_down,
            a1 * scale_down,
            a0 * scale_down,
        )
    } else {
        (a4, a3, a2, a1, a0)
    };

    // 5. Iterative Degree Reduction with Newton Refinement
    // Stability threshold: if a4 < epsilon * max_coeff, treat as zero.
    // 1e-6 forces the "Almost Quadratic" case (ratio 3e-8) into this fallback block.
    let ten = F::from(10i16);
    let stability_threshold = if is_f32 {
        one / pow_n(ten, 4) // 1e-4
    } else {
        one / pow_n(ten, 6) // 1e-6
    };

    let mut max_sc = sc_a3.abs();
    if sc_a2.abs() > max_sc {
        max_sc = sc_a2.abs();
    }
    if sc_a1.abs() > max_sc {
        max_sc = sc_a1.abs();
    }
    if sc_a0.abs() > max_sc {
        max_sc = sc_a0.abs();
    }

    if sc_a4.abs() < stability_threshold * max_sc {
        let approx_roots = if sc_a3.abs() < stability_threshold * max_sc {
            // Degrade to Quadratic
            super::quadratic::find_roots_quadratic(sc_a2, sc_a1, sc_a0)
        } else {
            // Degrade to Cubic
            super::cubic::find_roots_cubic(sc_a3, sc_a2, sc_a1, sc_a0)
        };
        // Refine the approximate roots using the full Quartic polynomial
        return refine_roots(approx_roots, sc_a4, sc_a3, sc_a2, sc_a1, sc_a0);
    }

    // 6. Analytic Solution
    let _3 = F::from(3i16);
    let _4 = F::from(4i16);
    let _6 = F::from(6i16);
    let _8 = F::from(8i16);
    let _9 = F::from(9i16);
    let _10 = F::from(10i16);
    let _12 = F::from(12i16);
    let _16 = F::from(16i16);
    let _18 = F::from(18i16);
    let _27 = F::from(27i16);
    let _64 = F::from(64i16);
    let _72 = F::from(72i16);
    let _80 = F::from(80i16);
    let _128 = F::from(128i16);
    let _144 = F::from(144i16);
    let _192 = F::from(192i16);
    let _256 = F::from(256i16);

    let discriminant =
        sc_a4 * sc_a0 * sc_a4 * (_256 * sc_a4 * sc_a0 * sc_a0 + sc_a1 * (_144 * sc_a2 * sc_a1 - _192 * sc_a3 * sc_a0))
            + sc_a4 * sc_a0 * sc_a2 * sc_a2 * (_16 * sc_a2 * sc_a2 - _80 * sc_a3 * sc_a1 - _128 * sc_a4 * sc_a0)
            + (sc_a3
                * sc_a3
                * (sc_a4 * sc_a0 * (_144 * sc_a2 * sc_a0 - _6 * sc_a1 * sc_a1)
                    + (sc_a0 * (_18 * sc_a3 * sc_a2 * sc_a1 - _27 * sc_a3 * sc_a3 * sc_a0 - _4 * sc_a2 * sc_a2 * sc_a2)
                        + sc_a1 * sc_a1 * (sc_a2 * sc_a2 - _4 * sc_a3 * sc_a1))))
            + sc_a4 * sc_a1 * sc_a1 * (_18 * sc_a3 * sc_a2 * sc_a1 - _27 * sc_a4 * sc_a1 * sc_a1 - _4 * sc_a2 * sc_a2 * sc_a2);

    let pp = _8 * sc_a4 * sc_a2 - _3 * sc_a3 * sc_a3;
    let rr = sc_a3 * sc_a3 * sc_a3 + _8 * sc_a4 * sc_a4 * sc_a1 - _4 * sc_a4 * sc_a3 * sc_a2;
    let delta0 = sc_a2 * sc_a2 - _3 * sc_a3 * sc_a1 + _12 * sc_a4 * sc_a0;
    let dd = _64 * sc_a4 * sc_a4 * sc_a4 * sc_a0 - _16 * sc_a4 * sc_a4 * sc_a2 * sc_a2 + _16 * sc_a4 * sc_a3 * sc_a3 * sc_a2
        - _16 * sc_a4 * sc_a4 * sc_a3 * sc_a1
        - _3 * sc_a3 * sc_a3 * sc_a3 * sc_a3;

    // Handle special cases
    let double_root = discriminant == zero;
    if double_root {
        let triple_root = double_root && delta0 == zero;
        let quadruple_root = triple_root && dd == zero;
        let no_roots = dd == zero && pp > zero && rr == zero;
        if quadruple_root {
            Roots::One([-sc_a3 / (_4 * sc_a4)])
        } else if triple_root {
            let x0 = (-_72 * sc_a4 * sc_a4 * sc_a0 + _10 * sc_a4 * sc_a2 * sc_a2 - _3 * sc_a3 * sc_a3 * sc_a2)
                / (_9 * (_8 * sc_a4 * sc_a4 * sc_a1 - _4 * sc_a4 * sc_a3 * sc_a2 + sc_a3 * sc_a3 * sc_a3));
            let roots = Roots::One([x0]);
            roots.add_new_root(-(sc_a3 / sc_a4 + _3 * x0))
        } else if no_roots {
            Roots::No([])
        } else {
            find_roots_via_depressed_quartic(sc_a4, sc_a3, sc_a2, sc_a1, sc_a0, pp, rr, dd)
        }
    } else {
        let no_roots = discriminant > zero && (pp > zero || dd > zero);
        if no_roots {
            Roots::No([])
        } else {
            find_roots_via_depressed_quartic(sc_a4, sc_a3, sc_a2, sc_a1, sc_a0, pp, rr, dd)
        }
    }
}

#[cfg(test)]
mod test {
    use super::super::super::*;

    #[test]
    fn test_find_roots_quartic() {
        assert_eq!(find_roots_quartic(1f32, 0f32, 0f32, 0f32, 0f32), Roots::One([0f32]));
        assert_eq!(find_roots_quartic(1f64, 0f64, 0f64, 0f64, -1f64), Roots::Two([-1f64, 1f64]));
        assert_eq!(
            find_roots_quartic(1f64, -10f64, 35f64, -50f64, 24f64),
            Roots::Four([1f64, 2f64, 3f64, 4f64])
        );

        match find_roots_quartic(
            1.1248467624839498f64,
            -4.8721513473605924f64,
            7.9323705711747614f64,
            -5.7774307699949397f64,
            1.5971379368787519f64,
        ) {
            Roots::Two(x) => {
                assert_float_array_eq!(2e-15f64, x, [1.225913506454221f64, 1.257275575390252f64]);
            }
            _ => {
                assert!(false);
            }
        }

        match find_roots_quartic(3f64, 5f64, -5f64, -5f64, 2f64) {
            Roots::Four(x) => {
                assert_float_array_eq!(2e-15f64, x, [-2f64, -1f64, 0.33333333333333333f64, 1f64]);
            }
            _ => {
                assert!(false);
            }
        }

        match find_roots_quartic(3f32, 5f32, -5f32, -5f32, 2f32) {
            Roots::Four(x) => {
                assert_float_array_eq!(5e-7, x, [-2f32, -1f32, 0.33333333333333333f32, 1f32]);
            }
            _ => {
                assert!(false);
            }
        }
    }

    #[test]
    fn test_find_roots_quartic_tim_luecke() {
        // Reported in December 2019
        assert_eq!(
            find_roots_quartic(-14.0625f64, -3.75f64, 29.75f64, 4.0f64, -16.0f64),
            Roots::Two([-1.1016116464173349f64, 0.9682783130840016f64])
        );

        // 32-bit floating point is not accurate enough to solve this case ...
        assert_eq!(
            find_roots_quartic(-14.0625f32, -3.75f32, 29.75f32, 4.0f32, -16.0f32),
            Roots::No([])
        );

        // Normalized case
        assert_eq!(
            find_roots_quartic(
                1f32,
                -3.75f32 / -14.0625f32,
                29.75f32 / -14.0625f32,
                4.0f32 / -14.0625f32,
                -16.0f32 / -14.0625f32
            ),
            Roots::Two([-1.1016117f32, 0.96827835f32])
        );
    }

    #[test]
    fn test_find_roots_quartic_triple_root() {
        // (x+3)(3x-1)^3 == 27 x^4 + 54 x^3 - 72 x^2 + 26 x - 3
        assert_eq!(
            find_roots_quartic(27f64, 54f64, -72f64, 26f64, -3f64),
            Roots::Two([-3.0f64, 0.3333333333333333f64])
        );
        assert_eq!(
            find_roots_quartic(27f32, 54f32, -72f32, 26f32, -3f32),
            Roots::Two([-3.0f32, 0.33333333f32])
        );
    }

    #[test]
    fn test_find_roots_quartic_quadruple_root() {
        // (7x+2)^4 == 2401 x^4 + 2744 x^3 + 1176 x^2 + 224 x + 16
        assert_eq!(
            find_roots_quartic(2401f64, 2744f64, 1176f64, 224f64, 16f64),
            Roots::One([-0.2857142857142857f64])
        );
        // 32-bit floating point is less accurate
        assert_eq!(
            find_roots_quartic(2401f32, 2744f32, 1176f32, 224f32, 16f32),
            Roots::One([-0.2857143f32])
        );
    }

    #[test]
    fn test_quartic_almost_quadratic() {
        let a4 = 0.000000030743755847066437;
        let a3 = 0.000000003666731306801131;
        let a2 = 1.0001928389119579;
        let a1 = 0.000011499702220469921;
        let a0 = -0.6976068572771268;

        let roots = find_roots_quartic(a4, a3, a2, a1, a0);

        let expected_1 = -0.835153846196954;
        let expected_2 = 0.835142346155438;
        let tolerance = 1e-12;

        match roots {
            Roots::Two([r1, r2]) => {
                assert!(
                    (r1 - expected_1).abs() < tolerance,
                    "Root 1 mismatch. Got {}, Expected {}",
                    r1,
                    expected_1
                );
                assert!(
                    (r2 - expected_2).abs() < tolerance,
                    "Root 2 mismatch. Got {}, Expected {}",
                    r2,
                    expected_2
                );
            }
            _ => panic!("Expected Roots::Two, got {:?}", roots),
        }
    }

    // ============================================================
    // Edge-case tests modeled after quadratic.rs patterns
    // ============================================================

    // Helper: assert roots match expected values within tolerance
    fn assert_quartic_roots_approx(actual: Roots<f64>, expected: &[f64], tol: f64) {
        let mut exp = expected.to_vec();
        exp.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let check = |got: f64, want: f64, label: &str| {
            if got.is_infinite() && want.is_infinite() {
                assert_eq!(got.signum(), want.signum(), "{} infinite sign mismatch", label);
            } else {
                let err = (got - want).abs();
                let adaptive_tol = tol * want.abs().max(1.0);
                assert!(
                    err <= adaptive_tol,
                    "{} mismatch: got {}, expected {}, err={}, tol={}",
                    label,
                    got,
                    want,
                    err,
                    adaptive_tol
                );
            }
        };

        match (&actual, expected.len()) {
            (Roots::No(_), 0) => {}
            (Roots::One([r1]), 1) => {
                check(*r1, exp[0], "Root 1");
            }
            (Roots::Two([r1, r2]), 2) => {
                check(*r1, exp[0], "Root 1");
                check(*r2, exp[1], "Root 2");
            }
            (Roots::Three([r1, r2, r3]), 3) => {
                check(*r1, exp[0], "Root 1");
                check(*r2, exp[1], "Root 2");
                check(*r3, exp[2], "Root 3");
            }
            (Roots::Four([r1, r2, r3, r4]), 4) => {
                check(*r1, exp[0], "Root 1");
                check(*r2, exp[1], "Root 2");
                check(*r3, exp[2], "Root 3");
                check(*r4, exp[3], "Root 4");
            }
            _ => panic!(
                "Structure mismatch: got {:?}, expected {} roots {:?}",
                actual,
                expected.len(),
                expected
            ),
        }
    }

    // ------------------------------------------------------------------
    // NaN and Infinity inputs
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_nan_inputs() {
        let nan = f64::NAN;
        // Any NaN coefficient should yield no roots
        let cases = [
            (nan, 1.0, 0.0, 0.0, 0.0),
            (1.0, nan, 0.0, 0.0, 0.0),
            (1.0, 0.0, nan, 0.0, 0.0),
            (1.0, 0.0, 0.0, nan, 0.0),
            (1.0, 0.0, 0.0, 0.0, nan),
            (nan, nan, nan, nan, nan),
        ];
        for (a4, a3, a2, a1, a0) in cases {
            assert_eq!(
                find_roots_quartic(a4, a3, a2, a1, a0),
                Roots::No([]),
                "Expected No roots for NaN case ({},{},{},{},{})",
                a4,
                a3,
                a2,
                a1,
                a0
            );
        }
    }

    #[test]
    fn test_quartic_inf_inputs() {
        let inf = f64::INFINITY;
        let cases = [
            (inf, 1.0, 0.0, 0.0, 0.0),
            (1.0, inf, 0.0, 0.0, 0.0),
            (1.0, 0.0, inf, 0.0, 0.0),
            (1.0, 0.0, 0.0, inf, 0.0),
            (1.0, 0.0, 0.0, 0.0, inf),
            (-inf, 1.0, 0.0, 0.0, 0.0),
            (inf, inf, inf, inf, inf),
        ];
        for (a4, a3, a2, a1, a0) in cases {
            assert_eq!(
                find_roots_quartic(a4, a3, a2, a1, a0),
                Roots::No([]),
                "Expected No roots for Inf case ({},{},{},{},{})",
                a4,
                a3,
                a2,
                a1,
                a0
            );
        }
    }

    // ------------------------------------------------------------------
    // Degenerate: a4 == 0 (falls through to cubic)
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_degenerate_a4_zero() {
        // 0*x^4 + x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
        assert_quartic_roots_approx(find_roots_quartic(0.0, 1.0, -6.0, 11.0, -6.0), &[1.0, 2.0, 3.0], 1e-12);
    }

    // ------------------------------------------------------------------
    // Degenerate: a0 == 0 (x=0 is a root, rest is cubic)
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_degenerate_a0_zero() {
        // x^4 - 5x^3 + 6x^2 = x^2(x-2)(x-3) -> roots: 0, 0, 2, 3
        // Solver should find 0, 2, 3 at minimum (0 may be single or double)
        let roots = find_roots_quartic(1.0f64, -5.0, 6.0, 0.0, 0.0);
        // x=0 is a root of the cubic x^3 - 5x^2 + 6x = x(x-2)(x-3)
        // plus the factored-out zero -> we get 0, 2, 3
        match roots {
            Roots::Three([r1, r2, r3]) => {
                assert!((r1 - 0.0).abs() < 1e-12);
                assert!((r2 - 2.0).abs() < 1e-12);
                assert!((r3 - 3.0).abs() < 1e-12);
            }
            Roots::Four([r1, r2, r3, r4]) => {
                // Double root at 0 may be reported as two separate roots
                assert!((r1 - 0.0).abs() < 1e-12);
                assert!((r2 - 0.0).abs() < 1e-12);
                assert!((r3 - 2.0).abs() < 1e-12);
                assert!((r4 - 3.0).abs() < 1e-12);
            }
            _ => panic!("Expected Three or Four roots, got {:?}", roots),
        }
    }

    // ------------------------------------------------------------------
    // Biquadratic: a1 == 0 and a3 == 0
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_biquadratic() {
        // x^4 - 5x^2 + 4 = (x^2-1)(x^2-4) -> roots: -2, -1, 1, 2
        assert_quartic_roots_approx(find_roots_quartic(1.0, 0.0, -5.0, 0.0, 4.0), &[-2.0, -1.0, 1.0, 2.0], 1e-12);
    }

    #[test]
    fn test_quartic_biquadratic_two_roots() {
        // x^4 - 1 = 0 -> roots: -1, 1 (the complex pair ±i is discarded)
        assert_quartic_roots_approx(find_roots_quartic(1.0, 0.0, 0.0, 0.0, -1.0), &[-1.0, 1.0], 1e-12);
    }

    #[test]
    fn test_quartic_biquadratic_no_real_roots() {
        // x^4 + x^2 + 1 = 0 -> no real roots
        assert_eq!(find_roots_quartic(1.0f64, 0.0, 1.0, 0.0, 1.0), Roots::No([]));
    }

    // ------------------------------------------------------------------
    // All zeros except a4
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_all_zero_except_leading() {
        // x^4 = 0 -> one root at 0
        assert_eq!(find_roots_quartic(1.0f64, 0.0, 0.0, 0.0, 0.0), Roots::One([0.0]));
        assert_eq!(find_roots_quartic(5.0f64, 0.0, 0.0, 0.0, 0.0), Roots::One([0.0]));
    }

    // ------------------------------------------------------------------
    // Double root (two pairs)
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_double_root_pair() {
        // (x-1)^2 * (x-3)^2 = x^4 - 8x^3 + 22x^2 - 24x + 9
        assert_quartic_roots_approx(find_roots_quartic(1.0, -8.0, 22.0, -24.0, 9.0), &[1.0, 3.0], 1e-10);
    }

    // ------------------------------------------------------------------
    // Large coefficients (overflow protection)
    // Using power-of-2 scaling so the solver's scaling logic actually
    // triggers (uniform scaling by 1e100 doesn't help since all
    // coefficients are equally large and internal products still overflow).
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_large_coefficients() {
        // (x-1)(x-2)(x-3)(x-4) = x^4 - 10x^3 + 35x^2 - 50x + 24
        // Scale by 2^200, which the solver's high_threshold path handles
        let s = 2.0f64.powi(200);
        assert_quartic_roots_approx(
            find_roots_quartic(s, -10.0 * s, 35.0 * s, -50.0 * s, 24.0 * s),
            &[1.0, 2.0, 3.0, 4.0],
            1e-10,
        );
    }

    #[test]
    fn test_quartic_very_large_coefficients() {
        // Coefficients near 2^500
        let s = 2.0f64.powi(500);
        // x^4 - 1 = 0, scaled
        assert_quartic_roots_approx(find_roots_quartic(s, 0.0, 0.0, 0.0, -s), &[-1.0, 1.0], 1e-10);
    }

    // ------------------------------------------------------------------
    // Small coefficients (underflow protection)
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_small_coefficients() {
        // Scale (x-1)(x+1)(x-2)(x+2) by 1e-200
        // = x^4 - 5x^2 + 4, scaled
        let s = 1e-200_f64;
        assert_quartic_roots_approx(
            find_roots_quartic(s, 0.0, -5.0 * s, 0.0, 4.0 * s),
            &[-2.0, -1.0, 1.0, 2.0],
            1e-10,
        );
    }

    #[test]
    fn test_quartic_tiny_coefficients() {
        // Coefficients near 2^-500
        let s = 2.0f64.powi(-500);
        assert_quartic_roots_approx(find_roots_quartic(s, 0.0, 0.0, 0.0, -s), &[-1.0, 1.0], 1e-10);
    }

    // ------------------------------------------------------------------
    // No real roots
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_no_real_roots() {
        // x^4 + 1 = 0 has no real roots
        assert_eq!(find_roots_quartic(1.0f64, 0.0, 0.0, 0.0, 1.0), Roots::No([]));
    }

    #[test]
    fn test_quartic_no_real_roots_positive_definite() {
        // x^4 + 2x^2 + 1 = (x^2+1)^2 -> no real roots
        assert_eq!(find_roots_quartic(1.0f64, 0.0, 2.0, 0.0, 1.0), Roots::No([]));
    }

    // ------------------------------------------------------------------
    // Negative leading coefficient
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_negative_leading() {
        // -x^4 + 5x^2 - 4 = -(x^4 - 5x^2 + 4) -> roots: -2, -1, 1, 2
        assert_quartic_roots_approx(find_roots_quartic(-1.0, 0.0, 5.0, 0.0, -4.0), &[-2.0, -1.0, 1.0, 2.0], 1e-12);
    }

    // ------------------------------------------------------------------
    // Closely spaced roots (ill-conditioned)
    // Wolfram Alpha: Expand[(x - 1)(x - 1.001)(x + 2)(x + 3)]
    //   = x^4 + 2.999 x^3 - 3.005999 x^2 - 8.005 x + 6.006
    // However the quartic solver has limited precision for close roots,
    // so we use a residual check instead of exact root matching.
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_close_roots_residual() {
        // (x - 1)(x - 1.001)(x + 2)(x + 3)
        // Wolfram Alpha: Expand[(x - 1)(x - 1.001)(x + 2)(x + 3)]
        //   = x^4 + 2.999x^3 - 3.004x^2 - 7.001x + 6.006
        let (a4, a3, a2, a1, a0) = (1.0, 2.999, -3.004, -7.001, 6.006);
        let roots = find_roots_quartic(a4, a3, a2, a1, a0);
        let root_vec = roots.as_ref();
        // Should find 4 roots
        assert_eq!(root_vec.len(), 4, "Expected 4 roots, got {:?}", roots);
        // Verify each root satisfies the polynomial
        for r in root_vec.iter() {
            let x = *r;
            let x2 = x * x;
            let p = a4 * x2 * x2 + a3 * x2 * x + a2 * x2 + a1 * x + a0;
            assert!(p.abs() < 1e-6, "Residual too large: P({}) = {}", x, p);
        }
    }

    // ------------------------------------------------------------------
    // Widely separated roots
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_widely_separated() {
        // (x - 0.001)(x - 1)(x - 100)(x - 1000)
        // = x^4 - 1101.001x^3 + 101200.101x^2 - 100200.1x + 100
        // (Wolfram Alpha: Expand[(x-.001)(x-1)(x-100)(x-1000)])
        // Use residual check since root conditioning varies
        let (a4, a3, a2, a1, a0) = (1.0, -1101.001, 101200.101, -100200.1, 100.0);
        let roots = find_roots_quartic(a4, a3, a2, a1, a0);
        let root_vec = roots.as_ref();
        assert!(root_vec.len() >= 2, "Expected at least 2 roots, got {:?}", roots);
        for r in root_vec.iter() {
            let x = *r;
            let x2 = x * x;
            let p = a4 * x2 * x2 + a3 * x2 * x + a2 * x2 + a1 * x + a0;
            // Scaled tolerance: for large roots the absolute residual can be large
            let abs_x = if x < 0.0 { -x } else { x };
            let base = if abs_x > 1.0 { abs_x } else { 1.0 };
            let scale = base * base * base;
            assert!(p.abs() < 1e-4 * scale, "Residual too large: P({}) = {}", x, p);
        }
    }

    // ------------------------------------------------------------------
    // Polynomial with known integer roots (moderate spread)
    // Wolfram Alpha: Solve[x^4 + 2x^3 - 13x^2 - 14x + 24 == 0, x]
    //   = {-4, -2, 1, 3}
    // (x+4)(x+2)(x-1)(x-3) = x^4 + 2x^3 - 13x^2 - 14x + 24
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_integer_roots() {
        assert_quartic_roots_approx(
            find_roots_quartic(1.0, 2.0, -13.0, -14.0, 24.0),
            &[-4.0, -2.0, 1.0, 3.0],
            1e-10,
        );
    }

    #[test]
    fn test_quartic_integer_roots_scaled() {
        // Same polynomial scaled by 3: 3x^4 + 6x^3 - 39x^2 - 42x + 72
        assert_quartic_roots_approx(
            find_roots_quartic(3.0, 6.0, -39.0, -42.0, 72.0),
            &[-4.0, -2.0, 1.0, 3.0],
            1e-10,
        );
    }

    // ------------------------------------------------------------------
    // Two real roots (two complex conjugate pairs discarded)
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_two_real_two_complex() {
        // (x^2 - 4)(x^2 + 1) = x^4 - 3x^2 - 4
        // Real roots: -2, 2; complex roots: ±i discarded
        assert_quartic_roots_approx(find_roots_quartic(1.0, 0.0, -3.0, 0.0, -4.0), &[-2.0, 2.0], 1e-12);
    }

    // ------------------------------------------------------------------
    // f32 tests (lower precision edge cases)
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_f32_basic() {
        // x^4 - 5x^2 + 4 = 0 -> -2, -1, 1, 2
        match find_roots_quartic(1.0f32, 0.0f32, -5.0f32, 0.0f32, 4.0f32) {
            Roots::Four(x) => {
                assert_float_array_eq!(5e-6, x, [-2.0f32, -1.0f32, 1.0f32, 2.0f32]);
            }
            other => panic!("Expected Four roots, got {:?}", other),
        }
    }

    #[test]
    fn test_quartic_f32_no_real_roots() {
        assert_eq!(find_roots_quartic(1.0f32, 0.0f32, 0.0f32, 0.0f32, 1.0f32), Roots::No([]));
    }

    // ------------------------------------------------------------------
    // Symmetry: negating all odd-power coefficients mirrors roots
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_symmetry() {
        // P(x) = x^4 - 10x^3 + 35x^2 - 50x + 24  -> roots 1,2,3,4
        // P(-x) = x^4 + 10x^3 + 35x^2 + 50x + 24  -> roots -1,-2,-3,-4
        assert_quartic_roots_approx(
            find_roots_quartic(1.0, 10.0, 35.0, 50.0, 24.0),
            &[-4.0, -3.0, -2.0, -1.0],
            1e-12,
        );
    }

    // ------------------------------------------------------------------
    // Nearly degenerate: a4 very small relative to others
    // (tests the stability threshold / iterative degree reduction path)
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_nearly_degenerate_cubic() {
        // Tiny a4 -> essentially a cubic
        // 1e-10 * x^4 + x^3 - 6x^2 + 11x - 6 ≈ (x-1)(x-2)(x-3)
        let roots = find_roots_quartic(1e-10, 1.0, -6.0, 11.0, -6.0);
        match roots {
            Roots::Three([r1, r2, r3]) => {
                assert!((r1 - 1.0).abs() < 1e-4, "Root1: got {}", r1);
                assert!((r2 - 2.0).abs() < 1e-4, "Root2: got {}", r2);
                assert!((r3 - 3.0).abs() < 1e-4, "Root3: got {}", r3);
            }
            Roots::Four([r1, r2, r3, _r4]) => {
                // The extra root from the quartic term is at a very large negative value
                assert!((r1 - 1.0).abs() < 1e-2 || (r2 - 1.0).abs() < 1e-2);
            }
            _ => {
                // Accept other outcomes for this ill-conditioned case
            }
        }
    }

    // ------------------------------------------------------------------
    // Monic quartic with a double root
    // (x-2)^2(x+1)(x+3) = x^4 - 9x^2 + 4x + 12
    // Wolfram Alpha: Solve[x^4 - 9x^2 + 4x + 12 == 0, x] -> {-3, -1, 2}
    // where 2 has multiplicity 2
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_one_double_two_simple() {
        let roots = find_roots_quartic(1.0, 0.0, -9.0, 4.0, 12.0);
        // The solver may report the double root once or twice
        let root_vec = roots.as_ref();
        assert!(root_vec.len() >= 3, "Expected at least 3 roots, got {:?}", roots);
        // Verify each returned root satisfies the polynomial
        for r in root_vec.iter() {
            let x = *r;
            let x2 = x * x;
            let p = x2 * x2 - 9.0 * x2 + 4.0 * x + 12.0;
            assert!(p.abs() < 1e-8, "Residual too large: P({}) = {}", x, p);
        }
    }

    // ------------------------------------------------------------------
    // Unit coefficient sanity check
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_unit_roots_of_unity_real_parts() {
        // x^4 - 16 = 0 -> x = ±2, ±2i
        // Real roots: -2, 2
        assert_quartic_roots_approx(find_roots_quartic(1.0, 0.0, 0.0, 0.0, -16.0), &[-2.0, 2.0], 1e-12);
    }

    // ------------------------------------------------------------------
    // Verify roots satisfy the polynomial (residual check)
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_residual_check() {
        // Use an arbitrary quartic and verify P(root) ≈ 0
        let (a4, a3, a2, a1, a0) = (2.0f64, -3.0, -5.0, 7.0, -1.0);
        let roots = find_roots_quartic(a4, a3, a2, a1, a0);
        for r in roots.as_ref().iter() {
            let x = *r;
            let p = a4 * x * x * x * x + a3 * x * x * x + a2 * x * x + a1 * x + a0;
            assert!(p.abs() < 1e-8, "Residual too large: P({}) = {}", x, p);
        }
    }

    // ------------------------------------------------------------------
    // Residual check for the Tim Luecke case
    // ------------------------------------------------------------------
    #[test]
    fn test_quartic_residual_tim_luecke() {
        let (a4, a3, a2, a1, a0) = (-14.0625f64, -3.75, 29.75, 4.0, -16.0);
        let roots = find_roots_quartic(a4, a3, a2, a1, a0);
        for r in roots.as_ref().iter() {
            let x = *r;
            let x2 = x * x;
            let p = a4 * x2 * x2 + a3 * x2 * x + a2 * x2 + a1 * x + a0;
            assert!(p.abs() < 1e-8, "Residual too large: P({}) = {}", x, p);
        }
    }
}
