"""Meridian's partner registry: a provider's name to the URL of its agent card.

The platform team added it so a seam can find a partner by name instead of a URL typed
into Claims' code. Partners register themselves through an onboarding form, so every
entry is a claim made by whoever filled the form in. A2A prescribes no registry API;
this one is Meridian's own, and as small as the incident needs.

A lookup returns candidates, newest first, so a partner that moves its card is found at
its new URL. A candidate is where to fetch a card, never a decision to trust it.
"""

from dataclasses import dataclass, field


@dataclass
class Registry:
    entries: list[tuple[str, str]] = field(default_factory=list)  # (provider, url)

    def register(self, provider: str, url: str) -> "Registry":
        self.entries.append((provider, url))
        return self

    def lookup(self, provider: str) -> list[str]:
        """Every card URL registered under this provider name, newest first."""
        return [url for name, url in reversed(self.entries) if name == provider]
