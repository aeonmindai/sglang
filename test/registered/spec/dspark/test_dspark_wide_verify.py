"""--speculative-dspark-verify-width and --speculative-dsv4-c4-ring on CPU: argument
parsing and refusals, the widths each side reads (the draft keeps gamma, verify takes
W), the proposer's extension columns, the one-hot draft distribution of an extension
column, and the default (W = gamma + 1) shapes."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.dspark.dspark_schedule import (
    schedule_verify_lens_topk,
)
from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.speculative_hook import (
    _handle_dspark,
    handle_speculative_decoding,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dspark_components.dspark_config import (
    DSparkDraftConfig,
    dspark_draft_gamma,
    resolve_dspark_verify_width,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftBlockResult,
    DraftProposal,
)
from sglang.srt.speculative.dspark_components.dspark_planner import (
    DSparkScheduleConfig,
    DSparkVerifyPlanner,
    admitted_chain_tokens,
    alloc_verify_window,
    uniform_ragged_layout,
)
from sglang.srt.speculative.dspark_components.dspark_verify import _draft_probs
from sglang.srt.speculative.ragged_verify import RaggedVerifyMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_HOOK = "sglang.srt.arg_groups.speculative_hook"
_CONFIG = "sglang.srt.speculative.dspark_components.dspark_config"
GAMMA = 5
WIDE = 17  # pick + extend at width 17: gamma 5 + 11 extension columns
VOCAB = 32


def _publish(testcase, **fields):
    override = get_context().override_server_args(**fields)
    override.install()
    testcase.addCleanup(override.restore)


def _parse(*flags) -> ServerArgs:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--model-path", "dummy", *flags])
    return ServerArgs.from_cli_args(args)


def _dspark_args(
    *,
    arch: str = "DeepseekV4ForCausalLM",
    compress_ratios=(0, 4, 128, 4),
    **fields,
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = "dummy-draft"
    server_args.speculative_dspark_block_size = GAMMA
    for name, value in fields.items():
        setattr(server_args, name, value)
    server_args._model_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            architectures=[arch], compress_ratios=list(compress_ratios)
        )
    )
    return server_args


def _declared_fields(server_args: ServerArgs) -> set:
    return {
        name
        for _, fields in getattr(server_args, "_resolved_overrides", [])
        for name in fields
    }


class TestFlagParsing(CustomTestCase):
    def test_defaults_keep_todays_verify(self):
        args = _parse()
        self.assertIsNone(args.speculative_dspark_verify_width)
        self.assertEqual(args.speculative_dsv4_c4_ring, 16)

    def test_values_parse_as_ints(self):
        args = _parse(
            "--speculative-dspark-verify-width",
            "17",
            "--speculative-dsv4-c4-ring",
            "32",
        )
        self.assertEqual(args.speculative_dspark_verify_width, 17)
        self.assertEqual(args.speculative_dsv4_c4_ring, 32)

    def test_malformed_values_are_refused(self):
        for flags in (
            ("--speculative-dsv4-c4-ring", "24"),
            ("--speculative-dsv4-c4-ring", "thirty-two"),
            ("--speculative-dspark-verify-width", "17.0"),
            ("--speculative-dspark-verify-width", "wide"),
        ):
            with self.subTest(flags=flags), self.assertRaises(SystemExit):
                _parse(*flags)


class TestResolveVerifyWidth(CustomTestCase):
    def test_default_is_gamma_plus_one(self):
        self.assertEqual(
            resolve_dspark_verify_width(
                gamma=GAMMA, verify_width=None, max_verify_width=10
            ),
            GAMMA + 1,
        )

    def test_width_must_hold_the_block(self):
        with self.assertRaisesRegex(ValueError, ">= gamma \\+ 1 = 6"):
            resolve_dspark_verify_width(
                gamma=GAMMA, verify_width=5, max_verify_width=None
            )

    def test_width_must_fit_the_ring(self):
        self.assertEqual(
            resolve_dspark_verify_width(
                gamma=GAMMA, verify_width=26, max_verify_width=26
            ),
            26,
        )
        with self.assertRaisesRegex(ValueError, "--speculative-dsv4-c4-ring"):
            resolve_dspark_verify_width(
                gamma=GAMMA, verify_width=27, max_verify_width=26
            )


class TestHandleDspark(CustomTestCase):
    def _handle(self, server_args: ServerArgs) -> ServerArgs:
        with patch(
            f"{_CONFIG}.read_draft_checkpoint_config",
            side_effect=OSError("no draft checkpoint on the CPU runner"),
        ):
            _handle_dspark(server_args)
        return server_args

    def test_default_resolves_gamma_plus_one_and_declares_nothing_new(self):
        server_args = self._handle(_dspark_args())
        self.assertEqual(
            resolution_result(server_args, "speculative_num_draft_tokens"), GAMMA + 1
        )
        self.assertNotIn("speculative_dspark_block_size", _declared_fields(server_args))

    def test_wide_width_on_ring_32(self):
        server_args = self._handle(
            _dspark_args(
                speculative_dspark_verify_width=WIDE, speculative_dsv4_c4_ring=32
            )
        )
        self.assertEqual(
            resolution_result(server_args, "speculative_num_draft_tokens"), WIDE
        )
        self.assertIn("speculative_dspark_block_size", _declared_fields(server_args))
        self.assertEqual(
            resolution_result(server_args, "speculative_dspark_block_size"), GAMMA
        )

    def test_ring_16_serves_width_10_only(self):
        server_args = self._handle(_dspark_args(speculative_dspark_verify_width=10))
        self.assertEqual(
            resolution_result(server_args, "speculative_num_draft_tokens"), 10
        )
        with self.assertRaisesRegex(ValueError, "--speculative-dsv4-c4-ring"):
            self._handle(_dspark_args(speculative_dspark_verify_width=11))

    def test_ring_32_serves_width_26_only(self):
        self._handle(
            _dspark_args(
                speculative_dspark_verify_width=26, speculative_dsv4_c4_ring=32
            )
        )
        with self.assertRaisesRegex(ValueError, "--speculative-dsv4-c4-ring"):
            self._handle(
                _dspark_args(
                    speculative_dspark_verify_width=27, speculative_dsv4_c4_ring=32
                )
            )

    def test_targets_without_the_c4_ring_take_any_width(self):
        for target in (
            {"arch": "LlamaForCausalLM", "compress_ratios": ()},
            # A V4 checkpoint without ratio-4 layers has no C4 ring to bound W.
            {"compress_ratios": (0, 0, 2, 1)},
        ):
            with self.subTest(**target):
                server_args = self._handle(
                    _dspark_args(speculative_dspark_verify_width=40, **target)
                )
                self.assertEqual(
                    resolution_result(server_args, "speculative_num_draft_tokens"), 40
                )

    def test_width_below_the_block_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cannot hold DSpark's block"):
            self._handle(_dspark_args(speculative_dspark_verify_width=GAMMA))

    def test_num_draft_tokens_must_match_the_width(self):
        with self.assertRaisesRegex(ValueError, "--speculative-dspark-verify-width"):
            self._handle(
                _dspark_args(
                    speculative_dspark_verify_width=10, speculative_num_draft_tokens=6
                )
            )

    def test_width_needs_a_known_gamma(self):
        with self.assertRaisesRegex(ValueError, "needs the draft block size"):
            self._handle(
                _dspark_args(
                    speculative_dspark_block_size=None,
                    speculative_num_draft_tokens=10,
                    speculative_dspark_verify_width=10,
                )
            )

    def test_width_is_dspark_only(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.speculative_algorithm = "DFLASH"
        server_args.speculative_dspark_verify_width = 10
        with (
            patch(f"{_HOOK}.run_post_process_pass"),
            patch(
                f"{_HOOK}._resolve_speculative_algorithm_alias", return_value="DFLASH"
            ),
            self.assertRaisesRegex(ValueError, "DSPARK only"),
        ):
            handle_speculative_decoding(server_args)


class TestDraftAndVerifyWidths(CustomTestCase):
    """The draft side keeps gamma rows; the verify side takes W."""

    def test_default_draft_rows_are_width_minus_one(self):
        _publish(self, speculative_algorithm="DSPARK", speculative_num_draft_tokens=6)
        self.assertEqual(dspark_draft_gamma(6), GAMMA)
        algo = SpeculativeAlgorithm.DSPARK
        self.assertEqual(
            algo.get_num_tokens_per_req_for_target_verify(6, is_draft_worker=True),
            GAMMA,
        )
        self.assertEqual(
            algo.get_num_tokens_per_req_for_target_verify(6, is_draft_worker=False), 6
        )

    def test_wide_draft_rows_stay_gamma(self):
        _publish(
            self,
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=WIDE,
            speculative_dspark_block_size=GAMMA,
            speculative_dspark_verify_width=WIDE,
        )
        self.assertEqual(dspark_draft_gamma(WIDE), GAMMA)
        algo = SpeculativeAlgorithm.DSPARK
        self.assertEqual(
            algo.get_num_tokens_per_req_for_target_verify(WIDE, is_draft_worker=True),
            GAMMA,
        )
        self.assertEqual(
            algo.get_num_tokens_per_req_for_target_verify(WIDE, is_draft_worker=False),
            WIDE,
        )

    def _runtime_config(self, num_draft_tokens: int):
        draft_config = DSparkDraftConfig(
            num_hidden_layers=3,
            num_target_layers=None,
            gamma=GAMMA,
            target_layer_ids=None,
            mask_token="<mask>",
            mask_token_id=7,
            markov_rank=256,
            markov_head_type="vanilla",
        )
        with patch(f"{_CONFIG}.parse_dspark_draft_config", return_value=draft_config):
            return resolve_runtime_config(
                draft_hf_config=None,
                speculative_num_draft_tokens=num_draft_tokens,
                target_vocab_size=VOCAB,
            )

    def test_runtime_config(self):
        _publish(self, speculative_algorithm="DSPARK", speculative_num_draft_tokens=6)
        default = self._runtime_config(6)
        self.assertEqual((default.gamma, default.verify_num_draft_tokens), (GAMMA, 6))
        _publish(
            self,
            speculative_algorithm="DSPARK",
            speculative_num_draft_tokens=WIDE,
            speculative_dspark_block_size=GAMMA,
            speculative_dspark_verify_width=WIDE,
        )
        wide = self._runtime_config(WIDE)
        self.assertEqual((wide.gamma, wide.verify_num_draft_tokens), (GAMMA, WIDE))

    def test_verify_window_and_layout_are_w_wide(self):
        bs = 3
        for width in (GAMMA + 1, WIDE):
            with self.subTest(width=width):
                plans = []

                def plan_uniform_extend(*, width, **_):
                    plans.append(width)
                    return SimpleNamespace(write_virtual=torch.arange(bs * width))

                window = alloc_verify_window(
                    batch=SimpleNamespace(
                        seq_lens=torch.tensor([10, 20, 30]),
                        req_pool_indices=torch.tensor([1, 2, 3]),
                    ),
                    bs=bs,
                    device="cpu",
                    verify_num_draft_tokens=width,
                    block_pos_offsets=torch.arange(width),
                    model_runner=SimpleNamespace(
                        runner_ids=SimpleNamespace(
                            plan_uniform_extend=plan_uniform_extend
                        )
                    ),
                    seq_lens_cpu=None,
                )
                self.assertEqual(plans, [width])
                self.assertEqual(tuple(window.positions_2d.shape), (bs, width))
                self.assertEqual(tuple(window.verify_cache_loc_2d.shape), (bs, width))
                self.assertEqual(window.positions_2d[2].tolist()[-1], 30 + width - 1)
                layout = uniform_ragged_layout(
                    bs=bs,
                    device=torch.device("cpu"),
                    verify_num_draft_tokens=width,
                    ragged_verify_mode=RaggedVerifyMode.COMPACT,
                    model_runner=SimpleNamespace(decode_cuda_graph_runner=None),
                )
                self.assertEqual(layout.verify_lens.tolist(), [width] * bs)
                self.assertEqual(layout.graph_num_tokens, bs * width)

    def test_out_tokens_span_the_wide_chain(self):
        draft_tokens = torch.arange(2 * (WIDE - 1)).view(2, WIDE - 1)
        out = BuildOutTokens.execute(
            draft_tokens=draft_tokens,
            correct_len=torch.tensor([3, WIDE - 1]),
            bonus=torch.tensor([100, 200]),
            verify_num_draft_tokens=WIDE,
            gamma=WIDE - 1,
        )
        self.assertEqual(tuple(out.shape), (2, WIDE))
        self.assertEqual(out[0, :3].tolist(), draft_tokens[0, :3].tolist())
        self.assertEqual(int(out[0, 3]), 100)
        self.assertEqual(out[1, : WIDE - 1].tolist(), draft_tokens[1].tolist())
        self.assertEqual(int(out[1, WIDE - 1]), 200)


class TestScheduleOverTheChain(CustomTestCase):
    def test_unfilled_extension_columns_are_never_scheduled(self):
        cfg = DSparkScheduleConfig(gamma=WIDE - 1)
        self.assertEqual(cfg.resolved_max_verify_len(), WIDE)
        confidence = torch.zeros(2, WIDE - 1)
        confidence[:, :GAMMA] = 0.9
        # Row 1: a plugin filled six extension columns.
        confidence[1, GAMMA : GAMMA + 6] = 0.95
        verify_lens = schedule_verify_lens_topk(
            confidence=confidence, budget=1000, cfg=cfg
        )
        self.assertEqual(verify_lens.tolist(), [GAMMA + 1, GAMMA + 7])

    def test_default_config_is_todays(self):
        cfg = DSparkScheduleConfig(gamma=(GAMMA + 1) - 1)
        self.assertEqual(cfg.resolved_max_verify_len(), GAMMA + 1)

    def test_admitted_chain_tokens_counts_filled_columns(self):
        cfg = DSparkScheduleConfig(gamma=WIDE - 1)
        self.assertIsNone(admitted_chain_tokens(None, cfg))
        # Row 0: DSpark's 5; row 1: DSpark's 5 plus 11 suffix tokens.
        self.assertEqual(admitted_chain_tokens(self._chain_confidence(), cfg), 21)

    @staticmethod
    def _chain_confidence() -> torch.Tensor:
        # Row 0: DSpark's block only; row 1: DSpark's block plus 11 suffix tokens.
        confidence = torch.zeros(2, WIDE - 1)
        confidence[:, :GAMMA] = 0.9
        confidence[1, GAMMA:] = 0.95
        return confidence

    def _verify_all_planner(self, verify_num_draft_tokens: int) -> DSparkVerifyPlanner:
        planner = DSparkVerifyPlanner.__new__(DSparkVerifyPlanner)
        planner.gamma = GAMMA
        planner.verify_num_draft_tokens = verify_num_draft_tokens
        planner.model_runner = SimpleNamespace(
            decode_cuda_graph_runner=SimpleNamespace(
                ragged_verify_mode=True, capture_num_tokens=[8, 16, 32, 64], max_bs=8
            )
        )
        planner._ragged_verify_mode = RaggedVerifyMode.COMPACT
        planner._schedule_cfg = DSparkScheduleConfig(gamma=verify_num_draft_tokens - 1)
        planner._budget_planner = object()
        planner._tp_sync = SimpleNamespace(sync=lambda site, tensor: None)
        planner._dynamic_graph_tier = True
        planner._is_verify_all = True
        planner._align_verify_tokens_to_graph_tier = False
        planner._uniform_layout_cache = {}
        return planner

    def _layout(self, planner, confidence):
        return planner.schedule_layout(
            req_pool_indices=torch.tensor([1, 2]),
            prefix_lens=torch.tensor([10, 20]),
            device=torch.device("cpu"),
            confidence=confidence,
            budget=None,
        )

    def test_wide_verify_all_takes_each_requests_chain(self):
        """Ragged verify-all with a wide chain: each request verifies its own filled
        chain (pick-one keeps DSpark's 6 rows, pick + extend 17), not W each."""
        layout = self._layout(self._verify_all_planner(WIDE), self._chain_confidence())
        self.assertEqual(layout.verify_lens.tolist(), [GAMMA + 1, WIDE])
        # 23 tokens on the graph grid, not 2 * 17 = 34 rounded up to 64.
        self.assertEqual(layout.graph_num_tokens, 32)

    def test_default_verify_all_keeps_the_uniform_layout(self):
        confidence = torch.full((2, GAMMA), 0.9)
        layout = self._layout(self._verify_all_planner(GAMMA + 1), confidence)
        self.assertEqual(layout.verify_lens.tolist(), [GAMMA + 1] * 2)
        self.assertEqual(layout.graph_num_tokens, 16)


def _block(bs: int, *, chain: int, greedy_draft: bool = False) -> DraftBlockResult:
    torch.manual_seed(0)
    return DraftBlockResult(
        draft_tokens=torch.randint(0, VOCAB, (bs, chain)),
        corrected_logits=None if greedy_draft else torch.randn(bs, GAMMA, VOCAB),
        greedy_mask=torch.zeros(bs, dtype=torch.bool),
        temperatures=torch.full((bs,), 0.7),
        greedy_draft=greedy_draft,
    )


class TestExtensionDraftProbs(CustomTestCase):
    def test_default_chain_is_the_softmax(self):
        block = _block(3, chain=GAMMA)
        probs = _draft_probs(block, VOCAB, "cpu")
        self.assertEqual(tuple(probs.shape), (3, GAMMA, VOCAB))
        torch.testing.assert_close(
            probs, torch.softmax(block.corrected_logits / 0.7, dim=-1)
        )

    def test_extension_columns_are_one_hot(self):
        block = _block(3, chain=WIDE - 1)
        probs = _draft_probs(block, VOCAB, "cpu")
        self.assertEqual(tuple(probs.shape), (3, WIDE - 1, VOCAB))
        torch.testing.assert_close(
            probs[:, :GAMMA], torch.softmax(block.corrected_logits / 0.7, dim=-1)
        )
        extension = block.draft_tokens[:, GAMMA:]
        self.assertTrue(
            torch.equal(
                probs[:, GAMMA:],
                torch.nn.functional.one_hot(extension, VOCAB).to(probs.dtype),
            )
        )

    def test_greedy_draft_is_one_hot_over_the_whole_chain(self):
        block = _block(2, chain=WIDE - 1, greedy_draft=True)
        probs = _draft_probs(block, VOCAB, "cpu")
        self.assertTrue(
            torch.equal(
                probs,
                torch.nn.functional.one_hot(block.draft_tokens, VOCAB).to(probs.dtype),
            )
        )


class TestProposerExtension(CustomTestCase):
    def _proposer(self, num_extension_tokens: int = 0) -> DraftBlockProposer:
        # The proposer resolves its logits layout from the published topology.
        self.enterContext(published_topology())
        kwargs = (
            {"num_extension_tokens": num_extension_tokens}
            if num_extension_tokens
            else {}
        )
        return DraftBlockProposer(
            draft_model=SimpleNamespace(sample_from_anchor=True),
            draft_model_runner=SimpleNamespace(device="cpu"),
            gamma=GAMMA,
            mask_token_id=1,
            draft_block_spec_info=None,
            tp_sync=None,
            random_seed=0,
            greedy_draft=False,
            **kwargs,
        )

    def _proposal(self, bs: int) -> DraftProposal:
        confidence = torch.rand(bs, GAMMA)
        return DraftProposal(
            draft_block_ids=torch.zeros(bs, GAMMA, dtype=torch.int64),
            draft_block=_block(bs, chain=GAMMA),
            confidence=confidence,
            confidence_raw=confidence.logit(),
            folded=True,
            verify_ids=torch.zeros(bs, GAMMA + 1, dtype=torch.int64),
        )

    def test_default_proposer_has_no_extension(self):
        self.assertEqual(self._proposer().num_extension_tokens, 0)

    def test_chain_is_dspark_block_then_reset_extension(self):
        proposer = self._proposer(num_extension_tokens=WIDE - 1 - GAMMA)
        epilogue_buf = torch.full((4 * (WIDE - 1),), -1, dtype=torch.int64)
        proposer.attach_chain_tokens_buf(epilogue_buf)
        bs = 3
        proposal = self._proposal(bs)
        wide = proposer._extend_chain(proposal, bs=bs)

        tokens = wide.draft_block.draft_tokens
        self.assertEqual(tuple(tokens.shape), (bs, WIDE - 1))
        self.assertEqual(tokens.data_ptr(), epilogue_buf.data_ptr())
        self.assertTrue(
            torch.equal(tokens[:, :GAMMA], proposal.draft_block.draft_tokens)
        )
        self.assertEqual(int(tokens[:, GAMMA:].abs().sum()), 0)
        self.assertIs(
            wide.draft_block.corrected_logits, proposal.draft_block.corrected_logits
        )
        self.assertEqual(tuple(wide.confidence.shape), (bs, WIDE - 1))
        self.assertTrue(torch.equal(wide.confidence[:, :GAMMA], proposal.confidence))
        self.assertEqual(float(wide.confidence[:, GAMMA:].abs().sum()), 0.0)
        self.assertIs(wide.confidence_raw, proposal.confidence_raw)
        self.assertIsNone(wide.verify_ids)
        self.assertTrue(wide.folded)

        # What a plugin wrote last step does not leak into the next one.
        tokens[:, GAMMA:] = 9
        wide.confidence[:, GAMMA:] = 0.9
        again = proposer._extend_chain(proposal, bs=bs)
        self.assertEqual(int(again.draft_block.draft_tokens[:, GAMMA:].abs().sum()), 0)
        self.assertEqual(float(again.confidence[:, GAMMA:].abs().sum()), 0.0)

    def test_batches_past_the_epilogue_buffer_get_their_own(self):
        proposer = self._proposer(num_extension_tokens=WIDE - 1 - GAMMA)
        epilogue_buf = torch.zeros((2 * (WIDE - 1),), dtype=torch.int64)
        proposer.attach_chain_tokens_buf(epilogue_buf)
        wide = proposer._extend_chain(self._proposal(3), bs=3)
        self.assertNotEqual(
            wide.draft_block.draft_tokens.data_ptr(), epilogue_buf.data_ptr()
        )
        self.assertEqual(tuple(wide.draft_block.draft_tokens.shape), (3, WIDE - 1))


if __name__ == "__main__":
    unittest.main()
