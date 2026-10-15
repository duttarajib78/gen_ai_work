Reference data (inputs). Never committed, never written to by any tool here, never baked into an image.

Place these four files in this folder (or in the folder named by DX_DATA_DIR):

  diagnosis_codes.tsv      headerless TSV, 8 columns, ICD-10-AM leaf codes (column 2 = ICD10_DIAGNOSIS, column 3 = code, column 7 = description)
  achi_codes.tsv           headerless TSV, 8 columns, ACHI procedure codes (column 2 = ICD10_PROCEDURE)
  MBS-XML-YYYYMMDD.XML     the Medicare Benefits Schedule XML; the newest file matching MBS-XML-*.XML is used
  all_pros.csv             prostheses billing codes (stacked extracts; header row: Product Category, Sub Category, Product Group,
                           Product Sub Group, Suffix, Billing Code, Benefit)

No licensed data to hand? `dx-support make-sample-data --out data-sample` writes tiny, invented, non-clinical files in the same formats, enough to run
every command with --dry-run and the HTTP service with the stub backend.

For deployment, deploy/scripts/package_model_data.sh packs these files into the SageMaker model artifact (model.tar.gz) and uploads it to the encrypted
artifacts bucket; SageMaker unpacks it to /opt/ml/model/data inside the container.
