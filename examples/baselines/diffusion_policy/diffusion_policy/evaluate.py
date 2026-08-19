from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from mani_skill.utils import common


def evaluate(
    n: int,
    agent,
    eval_envs,
    device,
    sim_backend: str,
    progress_bar: bool = True,
):
    agent.eval()

    if progress_bar:
        pbar = tqdm(total=n)

    eval_metrics = defaultdict(list)

    def add_episode_metrics(episode_info):
        """把单环境或并行环境的 episode 指标统一展开成一维列表。"""
        for k, v in episode_info.items():
            # Gymnasium vector env 可能生成一些内部 mask，例如 _return
            if k.startswith("_"):
                continue

            if torch.is_tensor(v):
                v = v.detach().float().cpu().numpy()
            else:
                v = np.asarray(v)

            if v.ndim == 0:
                eval_metrics[k].append(v.item())
            else:
                eval_metrics[k].extend(v.reshape(-1).tolist())

    with torch.no_grad():
        obs, info = eval_envs.reset()
        eps_count = 0

        while eps_count < n:
            obs = common.to_tensor(obs, device)

            # DP 一次生成一个 action chunk
            action_seq = agent.get_action(obs)

            if sim_backend == "physx_cpu":
                action_seq = action_seq.cpu().numpy()

            # 逐个执行 action chunk 中的动作
            for i in range(action_seq.shape[1]):
                obs, rew, terminated, truncated, info = eval_envs.step(
                    action_seq[:, i]
                )

                # episode 已结束，不继续执行剩余 action
                if truncated.any():
                    break

            # 在退出 action chunk 后统一统计本轮 episode
            if truncated.any():
                assert truncated.all() == truncated.any(), (
                    "all episodes should truncate at the same time "
                    "for fair evaluation with other algorithms"
                )

                # 兼容旧接口
                if "final_info" in info:
                    final_info = info["final_info"]

                    if isinstance(final_info, dict):
                        add_episode_metrics(final_info["episode"])
                    else:
                        for item in final_info:
                            if item is not None:
                                add_episode_metrics(item["episode"])

                # 兼容当前 ManiSkill + Gymnasium CPU wrapper
                elif "episode" in info:
                    add_episode_metrics(info["episode"])

                else:
                    raise KeyError(
                        "Evaluation info contains neither "
                        "'final_info' nor 'episode'. "
                        f"Available keys: {list(info.keys())}"
                    )

                eps_count += eval_envs.num_envs

                if progress_bar:
                    pbar.update(eval_envs.num_envs)

    if progress_bar:
        pbar.close()

    agent.train()

    for k in eval_metrics.keys():
        eval_metrics[k] = np.asarray(eval_metrics[k])

    return eval_metrics