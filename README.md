# Red neuronal simple

## Clonar repositorio
```
cd /workspace
git clone https://github.com/fernando-fetis/simple_training.git
cd simple_training
```

## Instalar dependencias

```
pip3 install -r requirements.txt
```

## Hacer el script ejecutable

Configurar `WANDB_API_KEY` en `run_pipeline.sh`. Luego:

```
chmod +x run_pipeline.sh
```

Sin embargo, Git guarda el bit de ejecución (el permiso `+x`) al versionar el archivo. Es decir, cuando se hace `git clone`, Git aplica ese permiso al archivo `run_pipeline.sh`, así que ya está listo para ejecutarse.

## Ejecutar entrenamiento e inferencia

```
./run_pipeline.sh
```

## Subir archivos a GitHub

```
git config user.name "Fernando Fetis"
git config user.email "ffetisriquelme@gmail.com"
git add .
git commit -m "cambios post-entrenamiento."
git push origin main
```