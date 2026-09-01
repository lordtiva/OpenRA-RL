# -*- coding: utf-8 -*-
"""Tests del traductor universal rl.roles (agnóstico a facción).

Verifica que:
  1) cada item del catalogo RA mapea a un rol (nada se pierde salvo known-misc)
  2) items equivalentes de distintas facciones (1TNK/2TNK/4TNK) comparten rol
  3) roles_de_available es estable y mapea rol -> items concretos disponibles
  4) mismo rol en dos facciones produce el mismo item para el rol (si ambas
     lo tienen) -> el indice que aprende la red NO depende de la facción.
"""
import sys
sys.path.insert(0, ".")

from rl import roles as R

ok = True
def check(name, cond):
    global ok
    print(f"  [{'OK' if cond else 'FALLA'}] {name}")
    ok = ok and bool(cond)

print("=== traductor universal rl.roles ===")

# 1) catálogo completo no debe tener huecos críticos: los items medidas
catalogo = ["e1", "e2", "e3", "e4", "e6", "dog", "spy", "med", "mcv", "harv",
            "v2rl", "1tnk", "2tnk", "3tnk", "4tnk", "qtnk", "stnk", "ctnk",
            "ttnk", "mgg", "arty", "apc", "truk", "mnly", "dtrk", "ftrk",
            "jeep", "mrj", "mig", "yak", "u2", "badr", "heli", "hind", "mh60",
            "tran", "ss", "msub", "dd", "ca", "pt", "lst",
            "proc", "silo", "powr", "apwr", "barr", "tent", "kenn", "weap",
            "dome", "atek", "stek", "gap", "gun", "agun", "pbox", "hbox",
            "ftur", "tsla", "sam", "fix", "hpad", "afld", "spen", "syrd",
            "fenc", "brik"]
no_huecos = all(R.role_of(it) != "misc" for it in catalogo)
check(f"{len(catalogo)} items del catalogo mapean (sin 'misc')", no_huecos)

# 2) equivalentes entre facciones comparten rol
check("1tnk/2tnk/4tnk son tanques (mismo rol tanque)",
      R.role_of("1tnk") == R.role_of("2tnk") == R.role_of("4tnk"))
check("e1 rifle y e2/e4 anti-infanteria",
      R.role_of("e2") == R.role_of("e4"))
check("e3 cohete es anti-blindaje (distinto de e1)",
      R.role_of("e3") != R.role_of("e1"))
check("role_id e1 != e3 y pad=0",
      R.role_id_of("e1") != R.role_id_of("e3")
      and R.ROLE_VOCAB["pad"] == 0
      and R.role_id_of("") == 0
      and R.role_id_of("no-existe") == R.ROLE_MISC_ID)

# 3) roles_de_available: estable y agrupa por rol
disp = {"1tnk", "proc", "powr", "e1", "ftrk", "apc", "2tnk", "harv"}
por_rol = R.roles_de_available(disp)
check("rol harvester presente", "harvester" in por_rol)
check("tanques 1tnk+2tnk agrupados bajo 'tank_light/medium'",
      "tank_light" in por_rol or "tank_medium" in por_rol)
check("e1 -> rol infantry_basic", "infantry_basic" in por_rol)
# el ítem concreto del rol tanque existe en la facción disponible
conj = set().union(*por_rol.values())
check("los roles disponibles solo usan items de la faccion (disponibles)",
      conj <= set(disp))

# 4) FACCIÓN A (aliados, 1tnk) vs FACCIÓN B (soviet, 2tnk): el ROL del tanque
#    NO cambia aunque el item concreto sea distinto.
rol_a = R.role_of("1tnk")
rol_b = R.role_of("2tnk")
check("mismo rol de tanque en ambas facciones (equivalencia funcional)",
      rol_a == rol_b)

check("pbox más barato que gun/agun",
      R.cheapest_of(["gun", "agun", "pbox"]) == "pbox")
check("ftur más barato que tsla",
      R.cheapest_of(["tsla", "ftur", "sam"]) == "ftur")

print("\n" + ("TODOS LOS TESTS OK" if ok else "HAY FALLAS ❌"))
sys.exit(0 if ok else 1)