from app.model.generator import Generator


class Wind(Generator):
    def __str__(self):
        production_sum = self["p_pv"].sum() if not self["p_pv"].empty else 0
        return f"Wind '{self.id}' with production sum = {production_sum}"
