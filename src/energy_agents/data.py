"""Customer, usage, tariff and outage data.

`InMemoryRepository` reads the bundled sample JSON for local runs and tests.
`DynamoDBRepository` reads the same entities from a single-table DynamoDB design:

| PK                 | SK                | Attributes                         |
|--------------------|-------------------|------------------------------------|
| CUSTOMER#<id>      | PROFILE           | name, area, plan, solar            |
| CUSTOMER#<id>      | USAGE#<YYYY-MM>   | kwh                                |
| TARIFF             | CURRENT           | tariff document (JSON)             |
| AREA#<area>        | OUTAGE#<ticket>   | status, reported_at, cause, ...    |
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str
    area: str
    plan: str
    solar: bool


class CustomerNotFoundError(KeyError):
    pass


class EnergyRepository(Protocol):
    def get_customer(self, customer_id: str) -> Customer: ...
    def get_monthly_usage(self, customer_id: str) -> dict[str, float]: ...
    def get_tariff(self) -> dict: ...
    def list_outages(self, area: str) -> list[dict]: ...


class InMemoryRepository:
    def __init__(self, document: dict) -> None:
        self._doc = document
        self._customers = {c["customer_id"]: c for c in document["customers"]}

    @classmethod
    def from_json(cls, path: str | Path) -> InMemoryRepository:
        return cls(json.loads(Path(path).read_text()))

    def _raw(self, customer_id: str) -> dict:
        try:
            return self._customers[customer_id]
        except KeyError:
            raise CustomerNotFoundError(customer_id) from None

    def get_customer(self, customer_id: str) -> Customer:
        raw = self._raw(customer_id)
        return Customer(raw["customer_id"], raw["name"], raw["area"], raw["plan"], raw["solar"])

    def get_monthly_usage(self, customer_id: str) -> dict[str, float]:
        return dict(sorted(self._raw(customer_id)["monthly_kwh"].items()))

    def get_tariff(self) -> dict:
        return self._doc["tariff"]

    def list_outages(self, area: str) -> list[dict]:
        return [o for o in self._doc["outages"] if o["area"].lower() == area.lower()]


class DynamoDBRepository:
    def __init__(self, table: Any) -> None:
        """`table` is a boto3 `dynamodb.Table` resource (or anything with the same methods)."""
        self._table = table

    @classmethod
    def from_name(cls, table_name: str, region: str | None = None) -> DynamoDBRepository:
        import boto3

        return cls(boto3.resource("dynamodb", region_name=region).Table(table_name))

    def _query(self, pk: str, sk_prefix: str) -> list[dict]:
        from boto3.dynamodb.conditions import Key

        items: list[dict] = []
        kwargs = {"KeyConditionExpression": Key("PK").eq(pk) & Key("SK").begins_with(sk_prefix)}
        while True:
            page = self._table.query(**kwargs)
            items.extend(page.get("Items", []))
            if "LastEvaluatedKey" not in page:
                return items
            kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]

    def get_customer(self, customer_id: str) -> Customer:
        item = self._table.get_item(Key={"PK": f"CUSTOMER#{customer_id}", "SK": "PROFILE"}).get("Item")
        if not item:
            raise CustomerNotFoundError(customer_id)
        return Customer(customer_id, item["name"], item["area"], item["plan"], bool(item["solar"]))

    def get_monthly_usage(self, customer_id: str) -> dict[str, float]:
        items = self._query(f"CUSTOMER#{customer_id}", "USAGE#")
        return dict(sorted((i["SK"].removeprefix("USAGE#"), float(i["kwh"])) for i in items))

    def get_tariff(self) -> dict:
        item = self._table.get_item(Key={"PK": "TARIFF", "SK": "CURRENT"})["Item"]
        return json.loads(item["document"])

    def list_outages(self, area: str) -> list[dict]:
        return self._query(f"AREA#{area}", "OUTAGE#")
