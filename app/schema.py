from pydantic import BaseModel, Field


class ProposedAssignment(BaseModel):
    """One truck for one delivery. Nothing else is proposable yet.

    The model names both as the table does (reference and plate): shorter than
    uuids and nothing to mistype. The proposer maps them back to ids and
    writes the title itself.
    """

    delivery: str = Field(description="Delivery reference from the table, e.g. DEL-2026-10001")
    truck: str = Field(description="Truck plate from the table, e.g. LAG-451-BC")
    why: str = Field(
        description="One sentence citing the table: distance to the pickup, "
        "how long the delivery has waited, how old the truck's position is."
    )


class TickOutput(BaseModel):
    proposals: list[ProposedAssignment] = Field(
        default_factory=list,
        description="Empty when nothing is clearly worth proposing.",
    )
