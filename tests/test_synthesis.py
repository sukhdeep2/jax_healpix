import pytest
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True) # Enable float64 for precision
import numpy as np # For healpy and test data

from jax_healpix.synthesis import MapSynthesizer
from jax_healpix.transform import HealpixTransformer # For analysis of output

# Test parameters
NSIDE_TEST = 2
LMAX_TEST = 2 * NSIDE_TEST
RAND_SEED = 42

@pytest.fixture
def synthesizer():
    return MapSynthesizer()

@pytest.fixture
def transformer_for_analysis():
    return HealpixTransformer(nside=NSIDE_TEST, l_max=LMAX_TEST)

def test_map_synthesizer_creation(synthesizer):
    assert synthesizer.transformer is None

    transformer = HealpixTransformer(nside=NSIDE_TEST, l_max=LMAX_TEST)
    synthesizer_with_trans = MapSynthesizer(healpix_transformer=transformer)
    assert synthesizer_with_trans.transformer is transformer

def test_synfast_temperature_only(synthesizer, transformer_for_analysis):
    nside = NSIDE_TEST
    lmax = LMAX_TEST

    cl_tt = jnp.ones(lmax + 1, dtype=jnp.float64) * 1e-4
    cl_tt = cl_tt.at[0:2].set(0.0)  # Ensure float64 for set value

    cls_input_for_synfast = cl_tt[None, :]

    tracer_info_t_only = {'T': {'cl_idx': 0}}

    maps_dict = synthesizer.synfast(
        nside=nside,
        l_max=lmax,
        spins_to_generate=(0,),
        tracer_info=tracer_info_t_only,
        cls_input=cls_input_for_synfast, # dtype is already float64
        rand_seed=RAND_SEED,
        healpix_transformer_instance=transformer_for_analysis
    )

    assert 0 in maps_dict
    map_T_2d = maps_dict[0]
    assert map_T_2d.shape == (4 * nside - 1, 4 * nside)

    alm_dict_out = transformer_for_analysis.map2alm(
        maps_dict={0: map_T_2d},
        spins_in_map=(0,)
    )
    alm_T_out = alm_dict_out[0]
    cl_tt_out = transformer_for_analysis.compute_cl(alm_T_out)

    assert cl_tt_out.shape == (lmax + 1,)
    assert not jnp.all(jnp.isclose(cl_tt_out[2:], jnp.array(0.0, dtype=jnp.float64)))

    mean_power_in = jnp.mean(cl_tt[2:])
    mean_power_out = jnp.mean(cl_tt_out[2:])
    assert jnp.isclose(mean_power_out, mean_power_in, rtol=1.5)


def test_synfast_polarization_e_b_from_separate_cls(synthesizer, transformer_for_analysis):
    nside = NSIDE_TEST
    lmax = LMAX_TEST

    cl_ee = jnp.ones(lmax + 1, dtype=jnp.float64) * 5e-5
    cl_ee = cl_ee.at[0:2].set(0.0)
    cl_bb = jnp.ones(lmax + 1, dtype=jnp.float64) * 2e-5
    cl_bb = cl_bb.at[0:2].set(0.0)
    cl_eb_zeros = jnp.zeros(lmax+1, dtype=jnp.float64)

    cls_input_for_synfast_pol = jnp.stack([cl_ee, cl_eb_zeros, cl_bb])

    tracer_info_pol = {
        'E': {'cl_idx': 0},
        'B': {'cl_idx': 1}
    }

    maps_dict_pol = synthesizer.synfast(
        nside=nside,
        l_max=lmax,
        spins_to_generate=(2, -2),
        tracer_info=tracer_info_pol,
        cls_input=cls_input_for_synfast_pol, # dtype is float64
        rand_seed=RAND_SEED,
        healpix_transformer_instance=transformer_for_analysis
    )

    assert 2 in maps_dict_pol
    assert -2 in maps_dict_pol
    map_Q_2d = maps_dict_pol[2]
    map_U_2d = maps_dict_pol[-2]

    assert map_Q_2d.shape == (4 * nside - 1, 4 * nside)
    assert map_U_2d.shape == (4 * nside - 1, 4 * nside)

    alm_dict_out_pol = transformer_for_analysis.map2alm(
        maps_dict={2: map_Q_2d, -2: map_U_2d},
        spins_in_map=(2, -2)
    )
    alm_E_out = alm_dict_out_pol[2]
    alm_B_out = alm_dict_out_pol[-2]

    cl_ee_out = transformer_for_analysis.compute_cl(alm_E_out)
    cl_bb_out = transformer_for_analysis.compute_cl(alm_B_out)
    cl_eb_out_dict = transformer_for_analysis.compute_cl({0: alm_E_out, 1: alm_B_out})
    cl_eb_out = cl_eb_out_dict[(0,1)]


    assert cl_ee_out.shape == (lmax + 1,)
    assert cl_bb_out.shape == (lmax + 1,)
    assert not jnp.all(jnp.isclose(cl_ee_out[2:], jnp.array(0.0, dtype=jnp.float64)))
    assert not jnp.all(jnp.isclose(cl_bb_out[2:], jnp.array(0.0, dtype=jnp.float64)))

    assert jnp.isclose(jnp.mean(cl_ee_out[2:]), jnp.mean(cl_ee[2:]), rtol=1.5)
    assert jnp.isclose(jnp.mean(cl_bb_out[2:]), jnp.mean(cl_bb[2:]), rtol=1.5)
    assert jnp.mean(jnp.abs(cl_eb_out[2:])) < 1e-5
