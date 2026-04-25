"""Defense components for the visual firewall pipeline."""

from src.defenses.cmc_firewall import CMCFirewall, CMCFirewallConfig, CMCSpanScore
from src.defenses.ocr_firewall import FirewallConfig, FirewallResult, apply_ocr_firewall
from src.defenses.semantic_firewall import SemanticFirewall, SemanticFirewallConfig

__all__ = [
    "FirewallConfig",
    "FirewallResult",
    "apply_ocr_firewall",
    "SemanticFirewall",
    "SemanticFirewallConfig",
    "CMCFirewall",
    "CMCFirewallConfig",
    "CMCSpanScore",
]
