@echo off

rem Ruta del directorio donde se encuentran los archivos de texto
set "directorio=D:\SEPTIMO SEMESTRE II\RECUPERACION INFORMACION\KevinMaldonado99\TASK 1\Books"

rem Palabra que deseas buscar
set "palabra_buscada=this"

rem Variables para contar el total de coincidencias
set "total_coincidencias=0"

rem Buscar la palabra en todos los archivos de texto en el directorio
for %%F in ("%directorio%\*.txt") do (
    for /f %%C in ('find /c /i "%palabra_buscada%" "%%F"') do (
        if %%C NEQ 0 (
            echo - Archivo: %%~nxF
            echo   Total de coincidencias: %%C
            set /a total_coincidencias+=%%C
        )
    )
)

rem Mostrar el total de coincidencias encontradas
echo Total de coincidencias encontradas: %total_coincidencias%

pause
