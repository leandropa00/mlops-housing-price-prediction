# Flujo de integración y despliegue

El proyecto mantiene el entrenamiento y las pruebas en GitHub Actions y la API de inferencia en Vercel. La implementación académica usa el check `tests-and-training` (workflow `pruebas-mlops`). Este documento explica las **configuraciones externas**: agregar los archivos al repositorio no activa por sí solo la protección.

## Flujo

1. Crear una rama de trabajo y un Pull Request a `main`.
2. GitHub Actions registra tres ejecuciones MLflow, comprueba las métricas y ejecuta Pytest.
3. Con `main` protegido, GitHub permite el merge solo si pasó `tests-and-training`.
4. Vercel genera el build de producción al actualizarse `main` y lo retiene hasta comprobar el mismo GitHub Check.
5. Vercel promociona el despliegue a producción automáticamente tras pasar el check.

## Activación en GitHub (requiere permisos de administración)

En https://github.com/leandropa00/mlops-housing-price-prediction/settings/branches :

- Crear una regla de protección para `main`.
- Activar `Require a pull request before merging`.
- Activar `Require status checks to pass before merging`.
- Elegir el check **`tests-and-training`** de GitHub Actions; seleccionar la fuente GitHub Actions si está disponible.
- Activar `Require branches to be up to date before merging` para validar el commit frente a la rama de destino.
- Activar `Do not allow bypassing the above settings` para aplicar la regla incluso a administradores.
- Desactivar force push y eliminación de rama.
- Guardar y comprobar que los PR sin pruebas satisfactorias no admiten merge.

Nota: exigir revisión de una persona adicional es opcional; puede bloquear un repositorio de un único propietario. Para el taller basta con PR + CI obligatorio.

## Activación en Vercel (requiere acceso al proyecto)

En Project Settings → Build and Deployment → Deployment Checks (la ubicación puede variar):

- Elegir **Add Checks → GitHub**.
- Seleccionar el check **`tests-and-training`** publicado por el workflow `pruebas-mlops` y marcarlo requerido para **Production**.
- Verificar que GitHub esté conectado y que en el entorno Production esté habilitada la asignación automática de dominios.
- No usar `Force Promote`: esa acción puede saltarse la compuerta.
- Comprobar en Deployments que el despliegue espera el resultado del check antes de promocionarse.

Guía oficial: https://vercel.com/docs/deployment-checks

## Validación práctica

Un PR normal debe generar un deployment Preview y el check `tests-and-training`. Una prueba fallida debe impedir el merge; una prueba exitosa permite fusionarlo. Tras fusionar, la nueva versión entra a producción **solo cuando el Deployment Check se completa satisfactoriamente**. Verificar el resultado de `GET /health` en el dominio público.

La creación del PR y el resultado del CI se pueden demostrar con enlaces a GitHub; la protección de rama y Deployment Checks deben verificarse en las respectivas consolas antes de declarar este control operativo.
