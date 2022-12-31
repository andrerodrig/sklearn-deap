def enum(**enums):
    return type("Enum", (), enums)


param_types = enum(Categorical=1, Numerical=2)
