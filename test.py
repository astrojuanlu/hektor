import toml

from hektor.guess import generate_initial_guess


with open("hektor.toml") as fp:
    conf = toml.load(fp)

print(generate_initial_guess(conf))
