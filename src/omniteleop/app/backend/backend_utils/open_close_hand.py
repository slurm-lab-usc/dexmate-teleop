# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to control robot hand movements.

Opens or closes both hands; choose the motion with the ``action`` CLI flag.
"""

import time
from typing import Literal

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main(action: Literal["open", "close"] = "close") -> None:
    """Open or close both robot hands, then shut down.

    Args:
        action: ``open`` or ``close`` for both hands.
    """
    logger.info("Initializing robot")
    with Robot(configure_default_state=False) as bot:
        if action == "open":
            logger.info("Opening both hands")
            bot.left_hand.open_hand()
            bot.right_hand.open_hand()
        else:
            logger.info("Closing both hands")
            bot.left_hand.close_hand()
            bot.right_hand.close_hand()

        logger.info("Waiting for hand movement to complete")
        time.sleep(2)


if __name__ == "__main__":
    tyro.cli(main)
